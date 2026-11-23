"""Bounded S2h shared-geometry verification; no VLA backbone or action policy."""
import argparse
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json, replace_json
from .probe_candidates import read, verify
from .audit import digest
from .geometry_readout import GeometryReadout, geometry_loss, load, prepare_learning
from .train_rotation_curriculum import state_hash


def train(run, device):
    p, q, inputs, labels = load(run)
    if not read(run / 'learning_preflight.json')['passed']: raise ValueError('Preflight required')
    if (run / 'learning_environment.json').exists(): raise ValueError('Fresh experiment required')
    x = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
    y = torch.from_numpy(labels['regression']).to(device); valid = torch.from_numpy(labels['regression_valid']).to(device)
    write_json(run / 'learning_environment.json', dict(torch=torch.__version__, device=device, threads=torch.get_num_threads(),
        gpu=None if device == 'cpu' else torch.cuda.get_device_name(0), deterministic=torch.are_deterministic_algorithms_enabled(),
        parameters=q['parameters'], backbone_loaded=False, simulator_loaded=False))
    start = time.monotonic(); reports = []
    def timeout(*_): raise TimeoutError('Geometry readout exceeded 600 seconds')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(600)
    try:
        for job in q['jobs']:
            weights = torch.tensor(job['weights'], dtype=torch.float32, device=device)
            for seed in q['seeds']:
                initial = next(v for v in q['initializations'] if v['scene'] == job['scene'] and v['seed'] == seed)
                model = GeometryReadout(p['channel_dim'], job['mean']).to(device)
                model.load_state_dict(torch.load(initial['path'], map_location=device, weights_only=True))
                assert state_hash(model) == initial['state_sha256']
                optimizer = torch.optim.AdamW(model.parameters(), lr=q['lr'], weight_decay=q['weight_decay'])
                history = []; tag = job['scene'].split(':')[-1] + '_' + job['condition'] + '_s' + str(seed)
                for step in range(1, q['updates'] + 1):
                    optimizer.zero_grad(); out = model(x); loss = geometry_loss(out['regression'], y, valid, weights)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                    loss.backward(); optimizer.step()
                    if step == 1 or step % 200 == 0:
                        history.append(dict(step=step, loss=float(loss.detach())))
                        replace_json(run / 'learning_progress.json', dict(tag=tag, update=step,
                            completed_models=len(reports), elapsed_seconds=time.monotonic() - start))
                model.eval(); checkpoint = run / (tag + '.pt'); torch.save(model.state_dict(), checkpoint)
                with torch.no_grad(): outputs = {k: v.cpu().numpy() for k, v in model(x).items()}
                artifact = run / (tag + '.npz'); np.savez_compressed(artifact, **outputs)
                reports.append(dict(scene=job['scene'], condition=job['condition'], seed=seed,
                    initial_sha256=initial['state_sha256'], updates=q['updates'], history=history,
                    checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), artifact=str(artifact), sha256=digest(artifact)))
                print('GEOMETRY_MODEL', tag, len(reports), 'loss', float(loss.detach()), flush=True)
        verify(p); verify(q)
        write_json(run / 'learning_result.json', dict(status='complete', reports=reports,
            elapsed_seconds=time.monotonic() - start,
            peak_gpu_allocated_bytes=None if device == 'cpu' else torch.cuda.max_memory_allocated(),
            scope='Shared diagnostic geometry only; no action adapter, task evaluation, router, TTA or new data collection.'))
        replace_json(run / 'learning_progress.json', dict(phase='complete', models=len(reports)))
        print('GEOMETRY_TRAINING_COMPLETE', run, flush=True)
    except BaseException:
        write_json(run / 'learning_failure.json', dict(completed_models=len(reports), traceback=traceback.format_exc())); raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('run', type=Path)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(); torch.set_num_threads(2); threadpool_limits(2); torch.use_deterministic_algorithms(True)
    if args.prepare: prepare_learning(args.run.resolve())
    else: train(args.run.resolve(), args.device)
