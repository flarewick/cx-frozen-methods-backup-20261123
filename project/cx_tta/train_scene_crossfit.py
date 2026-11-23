"""Bounded local S1y candidate training with fold-local preprocessing and loss."""
import argparse
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json, replace_json
from .audit import digest
from .probe_candidates import read, verify
from .scene_crossfit_data import prepare
from .time_encoding import TimeEncodedAdapter
from .pose_adapter import TorchFK
from .robot_kinematics import RobotFK
from .recovery_temporal import temporal_loss
from .train_semantic_pilot import balanced_weights
from .train_rotation_curriculum import state_hash


def end_torch(x, robots):
    return torch.stack([r(x[..., s:s+6]) for r, s in zip(robots, (0, 7))], 2)


def candidate_loss(pred, target, valid, weights, pairs, pair_weights, robots, caps):
    bc = (((pred - target) / caps).square().mean(-1) * valid).sum(-1) / valid.sum(-1)
    pe, te = end_torch(pred, robots), end_torch(target, robots)
    mask = valid[:, :, None].expand(-1, -1, 2)
    rot = ((pe[..., :3, :3] - te[..., :3, :3]).square().sum((-2, -1)) / 2 / .1**2)
    pos = (pe[..., :3, 3] - te[..., :3, 3]).square().sum(-1) / .02**2
    geometry = ((rot + pos) * mask).sum((1, 2)) / mask.sum((1, 2))
    within, seam = temporal_loss(pred, target, valid, caps, weights, pairs, pair_weights)
    return ((bc + geometry) * weights).sum() + within + seam


def pair_data(p, job, device):
    local = {g: i for i, g in enumerate(job['train'])}
    edges = [e for e in p['seam_pairs'] if e['before'] in local and e['after'] in local]
    for e in p['seam_pairs']:
        if (e['before'] in local) != (e['after'] in local):
            raise ValueError('Scene split broke a recovery boundary')
    index = torch.tensor([[local[e['before']], e['last'], local[e['after']]] for e in edges],
                         device=device, dtype=torch.long).reshape(-1, 3)
    weights = balanced_weights(edges).float().to(device) if edges else torch.empty(0, device=device)
    return index, weights


def interface_check(p, job, data, context, weights, robots, device):
    torch.manual_seed(19021)
    model = TimeEncodedAdapter(157, encoding='fourier', envelope=True).to(device)
    with torch.no_grad():
        model.head.weight.normal_(0, .01)
    ids = torch.tensor(job['train'], device=device)
    pairs, pw = pair_data(p, job, device)
    pred = model(context[ids], data['base'][ids])
    args = [data['target'][ids], data['valid'][ids], weights[ids], pairs, pw, robots, model.caps]
    loss = candidate_loss(pred, *args)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
    if not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
        raise ValueError('Missing real-data gradient')
    padded = args[0].clone()
    padded[~args[1]] = 9.25
    again = candidate_loss(pred, padded, *args[1:])
    gs = torch.autograd.grad(again, tuple(model.parameters()))
    difference = max(float((a - b).abs().max()) for a, b in zip(gradients, gs))
    if float((loss - again).abs()) > 1e-6 or difference > 1e-6:
        raise ValueError('Padded targets changed loss/gradient')
    training_groups = {p['rows'][i]['group'] for i in job['train']}
    predicted_groups = {p['rows'][i]['group'] for i in job['predict']}
    if training_groups & predicted_groups:
        raise ValueError('Candidate scene leakage')
    return dict(passed=True, valid_tail_loss_error=float((loss - again).abs().detach()),
                valid_tail_gradient_error=difference, real_loss=float(loss.detach()),
                train_scenes=len(training_groups), predicted_scenes=len(predicted_groups))


def train(run, device):
    p = read(run / 'protocol.json')
    verify(p)
    if (run / 'result.json').exists() or (run / 'environment.json').exists():
        raise ValueError('Fresh candidate training run required')
    with np.load(run / 'data.npz') as z:
        data = {k: torch.as_tensor(z[k], device=device) for k in ('base', 'target', 'valid')}
    robots = [TorchFK(RobotFK(read(Path(p['config'])), side)).to(device).float() for side in ('left', 'right')]
    write_json(run / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda, device=device,
        gpu=None if device == 'cpu' else torch.cuda.get_device_name(0), deterministic=torch.are_deterministic_algorithms_enabled()))
    reports, checks = [], []
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Crossfit candidate budget exceeded1800s')))
    signal.alarm(1800)
    try:
        for job in p['jobs']:
            with np.load(job['context']) as z:
                context = torch.as_tensor(z['context'], device=device)
                weights = torch.as_tensor(z['weights'], device=device)
            checks.append(dict(job=job['name'], **interface_check(p, job, data, context, weights, robots, device)))
            ids = torch.tensor(job['train'], device=device)
            pairs, pair_weights = pair_data(p, job, device)
            for encoding in p['encodings']:
                for seed in p['seeds']:
                    tag = f"{job['name']}_{encoding}_s{seed}"
                    torch.manual_seed(seed)
                    model = TimeEncodedAdapter(157, encoding=encoding, envelope=True).to(device)
                    if sum(v.numel() for v in model.parameters()) != p['parameters']:
                        raise ValueError('Changed candidate architecture')
                    initial = state_hash(model)
                    opt = torch.optim.AdamW(model.parameters(), lr=p['lr'])
                    history = []
                    for update in range(1, p['updates'] + 1):
                        opt.zero_grad()
                        pred = model(context[ids], data['base'][ids])
                        loss = candidate_loss(pred, data['target'][ids], data['valid'][ids], weights[ids],
                                              pairs, pair_weights, robots, model.caps)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite candidate loss')
                        loss.backward()
                        opt.step()
                        if update == 1 or update % 200 == 0:
                            record = dict(update=update, loss=float(loss.detach()))
                            history.append(record)
                            replace_json(run / 'progress.json', dict(phase='candidate_training', tag=tag,
                                **record, completed_models=len(reports), elapsed_seconds=time.monotonic()-started))
                    checkpoint = run / (tag + '.pt')
                    model.eval()
                    torch.save(model.state_dict(), checkpoint)
                    restored = TimeEncodedAdapter(157, encoding=encoding, envelope=True).to(device)
                    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                    restored.eval()
                    predicted = []
                    with torch.no_grad():
                        for start in range(0, len(job['predict']), 64):
                            ii = torch.tensor(job['predict'][start:start+64], device=device)
                            values = model(context[ii], data['base'][ii])
                            if not torch.equal(values, restored(context[ii], data['base'][ii])):
                                raise ValueError('Checkpoint reload mismatch')
                            predicted.append(values.cpu().numpy())
                    file = run / (tag + '.npz')
                    values = np.concatenate(predicted)
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite OOF prediction')
                    np.savez_compressed(file, predicted=values, indices=np.array(job['predict']))
                    write_json(run / (tag + '_history.json'), history)
                    reports.append(dict(job=job['name'], encoding=encoding, seed=seed, initial_sha256=initial,
                        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), file=str(file), sha256=digest(file),
                        training_rows=len(job['train']), predicted_rows=len(job['predict']), final_loss=float(loss.detach())))
                    print('CANDIDATE', tag, 'loss', float(loss.detach()), 'completed', len(reports), flush=True)
        verify(p)
        write_json(run / 'interface_checks.json', dict(passed=True, checks=checks))
        write_json(run / 'result.json', dict(status='complete', reports=reports, elapsed_seconds=time.monotonic()-started,
            peak_gpu_memory_bytes=None if device == 'cpu' else torch.cuda.max_memory_allocated(), scope=p['scope']))
        replace_json(run / 'progress.json', dict(phase='complete', models=len(reports)))
        print('COMPLETE', run, flush=True)
    except BaseException:
        write_json(run / 'failure.json', dict(traceback=traceback.format_exc(), completed_models=len(reports)))
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.use_deterministic_algorithms(True)
    if args.prepare_only:
        prepare()
    elif args.run is None:
        parser.error('--run required')
    else:
        train(args.run.resolve(), args.device)


if __name__ == '__main__':
    main()
