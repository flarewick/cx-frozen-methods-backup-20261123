"""S2j matched executable residual candidates; fixed final-checkpoint budget."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .audit import digest
from .paths import ROOT, output_path, write_json, replace_json
from .structured_action_adapter import StructuredActionAdapter, action_direction_loss, FORWARD_KEYS

SCOPE = 'S2j structured bounded direction residual'
COEFFICIENTS = {'S0': 0., 'SP': .1}


def read(path):
    return json.loads(Path(path).read_text())


def verify(sources):
    for path, sha in sources.items():
        if digest(Path(path)) != sha:
            raise ValueError('Source changed: ' + path)


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load(run):
    p = read(run / 'protocol.json')
    q = read(run / 'training_protocol.json') if (run / 'training_protocol.json').exists() else None
    verify(p['source_sha256'])
    if q is not None:
        verify(q['source_sha256'])
    arrays = []
    for name in ('inputs.npz', 'labels.npz'):
        with np.load(run / name, allow_pickle=False) as z:
            arrays.append({key: z[key].copy() for key in z.files})
    if set(arrays[0]) != FORWARD_KEYS:
        raise ValueError('Observable whitelist mismatch')
    return p, q, *arrays


def auxiliary_position_loss(pred, target, valid, weights):
    """XYZ numerator only, ORIGINAL full-24 valid denominator and row weights."""
    if pred.shape != target.shape or pred.shape != valid.shape or pred.shape[1] != 24:
        raise ValueError('Geometry shape mismatch')
    active = valid.bool().clone()
    active[:, 12:] = False
    active &= weights[:, None] > 0
    if not torch.isfinite(target[active]).all():
        raise ValueError('Nonfinite active geometry label')
    # Do not let invalid, disabled or zero-row NaNs poison gradients.
    safe = torch.where(active, target, pred.detach())
    error = (pred - safe).square() * active
    return ((error.sum(1) / valid.sum(1).clamp_min(1)) * weights).sum()


def losses(output, labels, caps, coefficient):
    action = action_direction_loss(output['delta'], labels['base'], labels['target'],
                                   labels['valid'], labels['row_weights'], caps)
    geometry = auxiliary_position_loss(output['geometry'], labels['geometry'],
                                      labels['geometry_valid'], labels['row_weights']) if coefficient else action.new_zeros(())
    return action + coefficient * geometry, action, geometry


def tensors(raw, device='cpu'):
    return {key: torch.from_numpy(value).to(device) for key, value in raw.items()}


def prepare(run):
    p, q, raw, labels = load(run)
    if q is not None or not read(run / 'preparation_audit.json')['passed']:
        raise ValueError('Fresh audited data run required')
    torch.manual_seed(0)
    model = StructuredActionAdapter(p['channel_dim'], labels['aux_mean'])
    initial = run / 'initial_seed0.pt'
    torch.save(model.state_dict(), initial)
    files = [run / name for name in ('protocol.json', 'preparation_audit.json', 'inputs.npz', 'labels.npz',
                                    'development_inputs.npz', 'development_labels.npz', 'initial_seed0.pt')]
    code = [Path(__file__), ROOT / 'cx_tta/structured_action_adapter.py', ROOT / 'cx_tta/geometry_readout.py',
            ROOT / 'cx_tta/structured_features.py', ROOT / 'cx_tta/prepare_structured_action.py',
            ROOT / 'scripts/train_structured_action.sh', ROOT / 'tests/test_structured_action_adapter.py']
    for f in code:
        (run / 'source_snapshot' / f.name).write_bytes(f.read_bytes())
    write_json(run / 'training_protocol.json', dict(scope=SCOPE, status='frozen_before_training',
        conditions=COEFFICIENTS, seed=0, updates=1400, lr=.0003, weight_decay=.01,
        batch='All611 rows, fixed order and original include_all weights on every update',
        channel_dim=p['channel_dim'], geometry_mean=labels['aux_mean'].tolist(),
        parameters=sum(v.numel() for v in model.parameters()), initial=str(initial),
        initial_state_sha256=state_hash(model), caps_native_units=model.caps.tolist(),
        action_loss='SmoothL1(delta/caps, clip((teacher-base)/caps,-1,1)); mean14 then valid-step rowmean; fixed original row weights. Raw teacher retained.',
        auxiliary_loss='SP .1 times squared XYZ0:12 numerator, original full24-valid per-row denominator and fixed original row weights; S0 coefficient0. Missing19 rows not fabricated.',
        deployment='Exactly9 legal observable fields + current native VLA base chunk; no geometry/valid/task/asset/seed/teacher phase inputs. Shared trunk, independent heads; geometry prediction not an action attractor.',
        checkpoint='Final1400 only; both conditions use identical actual initialization. No best seed/budget/threshold selection.',
        comparison='Closed-loop L0/S0/SP, seeds100380..100399, official compiled/internal noise/per-action observation. Development screening, not independent confirmation.',
        backbone_trainable_parameters=0, time_limit_seconds=600,
        source_sha256={str(f): digest(f) for f in files + code}))
    print('STRUCTURED_ACTION_PROTOCOL_FROZEN', run, flush=True)


def preflight(run):
    p, q, raw, arrays = load(run)
    if q is None:
        raise ValueError('Freeze protocol first')
    x, y = tensors(raw), tensors(arrays)
    initial = torch.load(q['initial'], map_location='cpu', weights_only=True)
    checks = []
    for condition, coefficient in COEFFICIENTS.items():
        model = StructuredActionAdapter(q['channel_dim'], q['geometry_mean'])
        model.load_state_dict(initial)
        assert state_hash(model) == q['initial_state_sha256']
        with torch.no_grad():
            result = model(x, y['base'])
            assert torch.equal(result['actions'], y['base']) and not result['delta'].any()
            assert torch.equal(model(x, y['base'], enabled=False)['actions'], y['base'])
        # Nonzero action output makes disabled/active label gradient tests meaningful.
        torch.manual_seed(901)
        with torch.no_grad():
            model.action_head[-1].weight.normal_(0, .01)
            model.action_head[-1].bias.normal_(0, .01)
        assert not any(isinstance(v, (torch.nn.Dropout, torch.nn.BatchNorm1d)) for v in model.modules())
        def calc(labels=y, inputs=x):
            loss = losses(model(inputs, labels['base']), labels, model.caps, coefficient)[0]
            gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
            return float(loss.detach()), [None if v is None else v.detach() for v in gradients]
        def difference(a, b):
            errors = [abs(a[0] - b[0])]
            for u, v in zip(a[1], b[1]):
                assert (u is None) == (v is None)
                if u is not None:
                    errors.append(float((u - v).abs().max()))
            return max(errors)
        reference = calc()
        yy = {k: v.clone() for k, v in y.items()}
        yy['target'][~yy['valid']] = float('nan')
        tail = difference(reference, calc(yy)); assert tail < 1e-6
        yy = {k: v.clone() for k, v in y.items()}
        yy['geometry'][:, 12:] = float('nan')
        yy['geometry'][~yy['geometry_valid']] = float('nan')
        disabled = difference(reference, calc(yy)); assert disabled < 1e-6
        if coefficient == 0:
            yy['geometry'][:] = float('nan')
            assert difference(reference, calc(yy)) < 1e-6
        anchor = next(i for i, row in enumerate(p['rows']) if row['kind'] == 'anchor')
        yy = {k: v.clone() for k, v in y.items()}
        yy['target'][anchor, 0, 0] += .02
        active = difference(reference, calc(yy)); assert active > 1e-6
        yy = {k: v.clone() for k, v in y.items()}
        yy['geometry'][0, 0] += .2
        auxiliary = difference(reference, calc(yy))
        assert auxiliary > 1e-6 if coefficient else auxiliary < 1e-6
        # Zero-weight row labels and observable inputs cannot affect either loss.
        yz = {k: v.clone() for k, v in y.items()}; yz['row_weights'][0] = 0
        zero_reference = calc(yz)
        yz['target'][0] = float('nan'); yz['geometry'][0] = float('nan')
        xz = {k: v.clone() for k, v in x.items()}
        for key, value in xz.items():
            value[0] = 0 if key == 'has_history' else 17.
        zero = difference(zero_reference, calc(yz, xz)); assert zero < 1e-6
        checks.append(dict(condition=condition, actual_initial_state_sha256=q['initial_state_sha256'],
            invalid_action_tail_gradient_error=tail, disabled_geometry_gradient_error=disabled,
            zero_weight_gradient_error=zero, active_action_gradient_change=active, auxiliary_gradient_change=auxiliary))
    verify(q['source_sha256'])
    write_json(run / 'training_preflight.json', dict(passed=True, checks=checks, gpu_used=False,
        source_sha256={str(run / 'training_protocol.json'): digest(run / 'training_protocol.json')}))
    print('STRUCTURED_ACTION_PREFLIGHT_PASSED', flush=True)


def metadata(run, p, q, condition):
    return dict(scope=SCOPE, condition=condition, channels=q['channel_dim'], geometry_mean=q['geometry_mean'],
        training_run=str(run), projection_path=p['projection'], projection_sha256=digest(Path(p['projection'])),
        robot_config_path=p['config'], robot_config_sha256=digest(Path(p['config'])),
        caps_native_units=q['caps_native_units'], updates=q['updates'], seed=q['seed'],
        parameters=q['parameters'], auxiliary_coefficient=COEFFICIENTS[condition],
        initial_state_sha256=q['initial_state_sha256'],
        training_protocol_sha256=digest(run / 'training_protocol.json'),
        action_target=q['action_loss'], auxiliary_target=q['auxiliary_loss'])


def train(run):
    p, q, raw, arrays = load(run)
    assert read(run / 'training_preflight.json')['passed']
    if (run / 'training_environment.json').exists():
        raise ValueError('Fresh training required; no silent resume')
    x, y = tensors(raw, 'cuda'), tensors(arrays, 'cuda')
    write_json(run / 'training_environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), deterministic=torch.are_deterministic_algorithms_enabled(),
        parameters=q['parameters'], backbone_loaded=False, simulator_loaded=False))
    start, reports = time.monotonic(), []
    def timeout(*_):
        raise TimeoutError('S2j exceeded fixed600seconds')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(q['time_limit_seconds'])
    try:
        for condition, coefficient in COEFFICIENTS.items():
            model = StructuredActionAdapter(q['channel_dim'], q['geometry_mean']).cuda()
            model.load_state_dict(torch.load(q['initial'], map_location='cuda', weights_only=True))
            assert state_hash(model) == q['initial_state_sha256']
            optimizer = torch.optim.AdamW(model.parameters(), lr=q['lr'], weight_decay=q['weight_decay'])
            history = []
            for step in range(1, q['updates'] + 1):
                optimizer.zero_grad(set_to_none=True)
                loss, action, auxiliary = losses(model(x, y['base']), y, model.caps, coefficient)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward(); optimizer.step()
                if step == 1 or step % 200 == 0:
                    entry = dict(step=step, loss=float(loss.detach()), action=float(action.detach()), auxiliary=float(auxiliary.detach()))
                    history.append(entry)
                    replace_json(run / 'training_progress.json', dict(condition=condition, completed_models=len(reports),
                        elapsed_seconds=time.monotonic()-start, **entry))
                    print(condition, entry, flush=True)
            model.eval()
            checkpoint = run / (condition + '_seed0.pt')
            torch.save(dict(model={k: v.cpu() for k, v in model.state_dict().items()},
                            metadata=metadata(run, p, q, condition)), checkpoint)
            with torch.inference_mode():
                predicted = {k: v.cpu().numpy() for k, v in model(x, y['base']).items()}
            artifact = run / (condition + '_predictions.npz')
            np.savez_compressed(artifact, **predicted)
            reports.append(dict(condition=condition, updates=q['updates'], initial_state_sha256=q['initial_state_sha256'],
                checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), predictions=str(artifact),
                predictions_sha256=digest(artifact), history=history))
        verify(p['source_sha256']); verify(q['source_sha256'])
        write_json(run / 'training_result.json', dict(status='complete', reports=reports,
            elapsed_seconds=time.monotonic()-start, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
        replace_json(run / 'training_progress.json', dict(phase='complete', models=len(reports)))
        print('STRUCTURED_ACTION_TRAINING_COMPLETE', run, flush=True)
    except BaseException:
        write_json(run / 'training_failure.json', dict(completed_models=len(reports), traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)


def audit(run):
    from .structured_action_deploy import read_checkpoint
    p, q, raw, arrays = load(run)
    result = read(run / 'training_result.json'); assert result['status'] == 'complete'
    x, y = tensors(raw), tensors(arrays)
    with np.load(run / 'development_inputs.npz') as z:
        xd = tensors({k: z[k].copy() for k in z.files})
    with np.load(run / 'development_labels.npz') as z:
        yd = tensors({k: z[k].copy() for k in z.files})
    reports = []
    for record in result['reports']:
        assert record['initial_state_sha256'] == q['initial_state_sha256'] and record['updates'] == q['updates']
        assert digest(Path(record['checkpoint'])) == record['checkpoint_sha256']
        assert digest(Path(record['predictions'])) == record['predictions_sha256']
        condition = record['condition']
        payload, _ = read_checkpoint(record['checkpoint'], condition)
        assert payload['metadata'] == metadata(run, p, q, condition)
        model = StructuredActionAdapter(q['channel_dim'], q['geometry_mean'])
        model.load_state_dict(payload['model']); model.eval()
        with torch.inference_mode():
            pred = model(x, y['base']); dev = model(xd, yd['base'])
            assert torch.equal(model(x, y['base'], enabled=False)['actions'], y['base'])
        with np.load(record['predictions']) as z:
            stored = {k: z[k].copy() for k in z.files}
        maximum = max(float(np.abs(pred[k].numpy()-stored[k]).max()) for k in pred)
        assert maximum < 2e-5 and all(np.isfinite(v).all() for v in stored.values())
        assert torch.all(pred['delta'].abs() <= model.caps + 1e-6)
        groups = {}
        for kind in sorted({r['kind'] for r in p['rows']}):
            ids = [i for i, r in enumerate(p['rows']) if r['kind'] == kind]
            valid = y['valid'][ids]
            original_weights = y['row_weights'][ids]
            w = original_weights / original_weights.sum()
            metrics = {}
            for tag, d in (('zero', torch.zeros_like(pred['delta'][ids])), ('candidate', pred['delta'][ids])):
                metrics[tag+'_direction_loss'] = float(action_direction_loss(d, y['base'][ids], y['target'][ids], valid, w, model.caps))
            rms = ((pred['actions'][ids]-y['target'][ids]).square().mean(-1)*valid).sum(-1)/valid.sum(-1).clamp_min(1)
            metrics['raw_teacher_native_rmse'] = float((rms*w).sum().sqrt())
            metrics['max_abs_correction'] = float(pred['delta'][ids].abs().max())
            groups[kind] = metrics
        dev_rms = (dev['delta'].square().mean(-1)*yd['valid']).sum(-1)/yd['valid'].sum(-1).clamp_min(1)
        reports.append(dict(condition=condition, cpu_reload_max_error=maximum, by_kind=groups,
            development_cup_anchor_native_rms=float(dev_rms.mean().sqrt()),
            development_cup_anchor_max_abs_correction=float(dev['delta'].abs().max())))
    write_json(run / 'training_audit.json', dict(passed=True, reports=reports,
        interpretation='Offline directional training fit and diagnostics only. No task-success improvement established; cup development anchors do not select weights or gate the bounded interface smoke.',
        source_sha256={str(run / 'training_result.json'): digest(run / 'training_result.json')}))
    print(json.dumps(dict(passed=True, reports=reports), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=('prepare', 'preflight', 'train', 'audit'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = output_path(args.run)
    torch.set_num_threads(2); threadpool_limits(2)
    torch.use_deterministic_algorithms(True)
    {'prepare': prepare, 'preflight': preflight, 'train': train, 'audit': audit}[args.phase](run)


if __name__ == '__main__':
    main()
