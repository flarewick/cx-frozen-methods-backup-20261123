"""S2l explicitly derived fixed-cache training: one row-weight recipe changes.

This entry consumes sealed arrays, not missing raw teacher/RGB files. It records
those inherited gaps rather than claiming their raw-source certificate passed.
The original S2j loader, checkpoints, source files and protocols stay immutable.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .audit import digest
from .paths import ROOT, new_run, output_path, replace_json, write_json
from .structured_action_adapter import FORWARD_KEYS, StructuredActionAdapter, action_direction_loss
from .train_structured_action import read, state_hash, tensors, verify

SCOPE = 'S2l explicit fixed-cache task-normalized action residual'
CONDITIONS = ('S0_control', 'S_task_balanced')
BASE_RUN = ROOT / 'outputs/effect_learning/structured_action_20261003_095511_385074'
DESIGN = ROOT / 'outputs/reports/attribution/components_accepted_20261004_100742/next_experiments.json'
DESIGN_SHA = 'ff8dc8266576b9a727b56ec0124e51ec8aa2e012def743b7df7de3c7ad96eea4'
TASKS = ('beat_block_hammer', 'place_shoe', 'turn_switch', 'place_empty_cup')


def normalized_weights(rows, original):
    if original.dtype != np.float32 or original.shape != (len(rows),) or not np.isfinite(original).all() or np.any(original <= 0):
        raise ValueError('Finite positive original float32 row weights required')
    names = np.array([row['task'] for row in rows])
    if set(names) != set(TASKS):
        raise ValueError('Exactly four registered training tasks required')
    for row in rows:
        try:
            scene = int(row['group'].rsplit(':', 1)[1])
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError('Explicit scene group required') from error
        if 100380 <= scene <= 100399:
            raise ValueError('Evaluation scene leaked into training')
        if row['kind'] not in ('expert', 'anchor', 'recovery'):
            raise ValueError('Unknown supervision kind')
    total = float(original.astype(np.float64).sum())
    result, factors = original.copy(), {}
    for task in TASKS:
        ids = names == task
        factor = (total / 4) / float(original[ids].astype(np.float64).sum())
        result[ids] = (original[ids].astype(np.float64) * factor).astype(np.float32)
        factors[task] = factor
    if abs(float(result.astype(np.float64).sum()) - total) > 1e-6:
        raise ValueError('Global supervision mass changed')
    return result, factors


def weight_summary(rows, values):
    groups = {}
    for field in ('task', 'kind'):
        groups[field] = {name: float(sum(float(values[i]) for i, row in enumerate(rows) if row[field] == name))
                         for name in sorted({row[field] for row in rows})}
    return dict(total_mass=float(values.astype(np.float64).sum()), **groups)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def parent_data(design):
    verify(design['source_sha256'])
    p, q = read(BASE_RUN / 'protocol.json'), read(BASE_RUN / 'training_protocol.json')
    # These are the exact consumed prepared arrays/code, not the 507 raw refs.
    verify(q['source_sha256'])
    missing = sorted(path for path in p['source_sha256'] if not Path(path).is_file())
    if missing != sorted(design['existing_source_gaps']['missing_sources']):
        raise ValueError('Inherited raw-source gaps differ from the declared design')
    if len(missing) != 33 or len(p['source_sha256']) != 507:
        raise ValueError('Unexpected inherited source coverage')
    x, y = arrays(BASE_RUN / 'inputs.npz'), arrays(BASE_RUN / 'labels.npz')
    if set(x) != FORWARD_KEYS or len(p['rows']) != 611 or len(x['state']) != 611:
        raise ValueError('Fixed 611-row observable cache required')
    if any(row['training_row'] != i for i, row in enumerate(p['rows'])):
        raise ValueError('Training row order changed')
    if (q['seed'] != 0 or q['updates'] != 1400 or q['lr'] != .0003 or q['weight_decay'] != .01
            or q['channel_dim'] != 32 or q['parameters'] != 42934):
        raise ValueError('Original matched training contract changed')
    expected, factors = normalized_weights(p['rows'], y['row_weights'])
    proposed = arrays(DESIGN.parent / 'proposed_row_weights.npz')
    if (set(proposed) != {'original', 'task_normalized'}
            or not np.array_equal(proposed['original'], y['row_weights'])
            or not np.array_equal(proposed['task_normalized'], expected)):
        raise ValueError('Preregistered weight arrays do not match the fixed recipe')
    return p, q, x, y, expected, factors


def prepare():
    if digest(DESIGN) != DESIGN_SHA:
        raise ValueError('Sealed S2l design changed')
    design = read(DESIGN)
    if design['status'] != 'prepared_not_executed' or design['stage'] != 'S2l':
        raise ValueError('Preregistered unexecuted S2l design required')
    p, q, x, y, balanced, factors = parent_data(design)
    from .structured_action_deploy import read_checkpoint
    original, _ = read_checkpoint(BASE_RUN / 'S0_seed0.pt', 'S0')
    model = StructuredActionAdapter(q['channel_dim'], q['geometry_mean'])
    model.load_state_dict(torch.load(q['initial'], map_location='cpu', weights_only=True), strict=True)
    if state_hash(model) != q['initial_state_sha256']:
        raise ValueError('Actual original initialization differs')
    model._validate(tensors(x), torch.from_numpy(y['base']))
    if not np.isfinite(y['target'][y['valid']]).all() or not y['valid'].sum(1).all():
        raise ValueError('Invalid active teacher cache')
    run = new_run('task_balanced_action', group=('effect_learning',))
    np.savez(run / 'row_weights.npz', S0_control=y['row_weights'], S_task_balanced=balanced)
    sources = dict(q['source_sha256'])
    sources.update(design['source_sha256'])
    for path in (DESIGN, BASE_RUN / 'S0_predictions.npz', Path(p['config']), Path(p['projection']), run / 'row_weights.npz'):
        sources[str(path)] = digest(path)
    code = [Path(__file__), ROOT / 'scripts/train_task_balanced_action.sh', ROOT / 'tests/test_task_balanced_action.py']
    snapshot = run / 'source_snapshot'; snapshot.mkdir()
    for path in code:
        (snapshot / path.name).write_bytes(path.read_bytes())
        sources[str(path)] = digest(path)
    write_json(run / 'training_protocol.json', dict(schema_version=1, scope=SCOPE,
        status='frozen_before_training', conditions=list(CONDITIONS), parent_run=str(BASE_RUN),
        design=str(DESIGN), design_sha256=DESIGN_SHA, data_mode='Explicit fixed-cache derivation',
        raw_teacher_revalidation_claim=False, inherited_reference_count=507, inherited_missing_sources=design['existing_source_gaps'],
        source_sha256=sources, initial=q['initial'], initial_state_sha256=q['initial_state_sha256'],
        seed=0, updates=1400, lr=q['lr'], weight_decay=q['weight_decay'], channel_dim=q['channel_dim'],
        parameters=q['parameters'], geometry_mean=q['geometry_mean'], caps_native_units=q['caps_native_units'],
        original_checkpoint_sha256=digest(BASE_RUN / 'S0_seed0.pt'),
        original_model_state_sha256=state_hash_from_state(original['model']),
        projection=p['projection'], robot_config=p['config'], row_weight_factors=factors,
        weight_summaries={CONDITIONS[0]:weight_summary(p['rows'], y['row_weights']), CONDITIONS[1]:weight_summary(p['rows'], balanced)},
        change='Only the sealed task-normalized row-weight recipe; no history/architecture/target/cap/auxiliary-loss/strength change',
        component_mass_confound_declared=True, action_loss=q['action_loss'], auxiliary_coefficient=0.,
        batch='All611 fixed-order rows at every update', checkpoint='Final1400, seed0 only',
        control_reproduction='Exact tensor equality to original S0 is required before treatment training',
        time_limit_seconds=600, backbone_trainable_parameters=0, evaluation=design['closed_loop_development']))
    print('RUN', run, flush=True)


def state_hash_from_state(state):
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load(run):
    q = read(run / 'training_protocol.json')
    if q['scope'] != SCOPE or q['conditions'] != list(CONDITIONS) or q['seed'] != 0 or q['updates'] != 1400:
        raise ValueError('Invalid derived paired training contract')
    verify(q['source_sha256'])
    p, old, x, y, balanced, _ = parent_data(read(DESIGN))
    weights = arrays(run / 'row_weights.npz')
    if set(weights) != set(CONDITIONS) or not np.array_equal(weights[CONDITIONS[0]], y['row_weights']) or not np.array_equal(weights[CONDITIONS[1]], balanced):
        raise ValueError('Derived row weights changed')
    return p, q, x, y, weights


def loss(model, x, y, weights):
    output = model(x, y['base'])
    # Preserve the original S0 total-loss expression as well as its gradients.
    action = action_direction_loss(output['delta'], y['base'], y['target'], y['valid'], weights, model.caps)
    return action + 0. * action.new_zeros(())


def preflight(run):
    p, q, raw, labels, weights = load(run)
    x, y = tensors(raw), tensors(labels)
    checks = []
    for condition in CONDITIONS:
        model = StructuredActionAdapter(q['channel_dim'], q['geometry_mean'])
        model.load_state_dict(torch.load(q['initial'], map_location='cpu', weights_only=True))
        assert state_hash(model) == q['initial_state_sha256']
        w = torch.from_numpy(weights[condition])
        with torch.no_grad():
            output = model(x, y['base'])
            assert torch.equal(output['actions'], y['base']) and not output['delta'].any()
        total = loss(model, x, y, w)
        total.backward()
        assert torch.isfinite(total) and model.action_head[-1].weight.grad.abs().max() > 0
        assert all(param.grad is None for param in model.geometry_head.parameters())
        model.zero_grad(set_to_none=True)
        # A full arbitrary geometry label perturbation must never enter S0 loss.
        altered = dict(y, geometry=torch.full_like(y['geometry'], float('nan')))
        assert torch.equal(loss(model, x, altered, w), total.detach())
        checks.append(dict(condition=condition, actual_initial_identity=True, identity_actions=True,
            geometry_excluded=True, finite_initial_loss=float(total.detach()),
            weights=weight_summary(p['rows'], weights[condition])))
    write_json(run / 'training_preflight.json', dict(passed=True, checks=checks, GPU_used=False,
        inherited_source_gaps_recorded=True, source_sha256={str(run / 'training_protocol.json'):digest(run / 'training_protocol.json')}))
    print('S2L_PREFLIGHT_PASSED', flush=True)


def metadata(run, q, condition):
    return dict(scope=SCOPE, condition=condition, training_run=str(run), channels=q['channel_dim'],
        geometry_mean=q['geometry_mean'], caps_native_units=q['caps_native_units'], parameters=q['parameters'],
        projection_path=q['projection'], projection_sha256=digest(Path(q['projection'])),
        robot_config_path=q['robot_config'], robot_config_sha256=digest(Path(q['robot_config'])),
        updates=1400, seed=0, initial_state_sha256=q['initial_state_sha256'], auxiliary_coefficient=0.,
        training_protocol_sha256=digest(run / 'training_protocol.json'), row_weights_sha256=digest(run / 'row_weights.npz'),
        parent_run=q['parent_run'], parent_checkpoint_sha256=q['original_checkpoint_sha256'],
        source_coverage='Fixed-cache derivation; raw source gaps explicitly retained')


def compare_control(state):
    reference = torch.load(BASE_RUN / 'S0_seed0.pt', map_location='cpu', weights_only=True)['model']
    if set(state) != set(reference):
        raise ValueError('Original control parameter structure differs')
    diff = {k:float((state[k].detach().cpu()-reference[k]).abs().max()) for k in reference}
    return dict(exact=all(torch.equal(state[k].detach().cpu(), reference[k]) for k in reference),
        max_abs=max(diff.values()), differences={k:v for k,v in diff.items() if v != 0},
        state_sha256=state_hash_from_state(state), original_state_sha256=state_hash_from_state(reference))


def train(run):
    p, q, raw, labels, weights = load(run)
    assert read(run / 'training_preflight.json')['passed']
    if (run / 'training_environment.json').exists():
        raise ValueError('Fresh training required; partial jobs are retained, not silently resumed')
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU visible: no training has started')
    x, y = tensors(raw, 'cuda'), tensors(labels, 'cuda')
    write_json(run / 'training_environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(), allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        parameters=q['parameters'], backbone_loaded=False, simulator_loaded=False))
    started, reports = time.monotonic(), []
    def timeout(*_):
        raise TimeoutError('Fixed600second micro-head budget exceeded')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(q['time_limit_seconds'])
    try:
        for condition in CONDITIONS:
            model = StructuredActionAdapter(q['channel_dim'], q['geometry_mean']).cuda()
            model.load_state_dict(torch.load(q['initial'], map_location='cuda', weights_only=True))
            assert state_hash(model) == q['initial_state_sha256']
            w = torch.from_numpy(weights[condition]).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=q['lr'], weight_decay=q['weight_decay'])
            history = []
            for step in range(1, q['updates']+1):
                optimizer.zero_grad(set_to_none=True)
                value = loss(model, x, y, w)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite loss')
                value.backward(); optimizer.step()
                if step == 1 or step % 200 == 0:
                    row = dict(step=step, loss=float(value.detach()))
                    history.append(row)
                    replace_json(run / 'training_progress.json', dict(condition=condition,
                        completed_models=len(reports), elapsed_seconds=time.monotonic()-started, **row))
                    print(condition, row, flush=True)
            model.eval()
            control = compare_control(model.state_dict()) if condition == CONDITIONS[0] else None
            if control is not None:
                write_json(run / 'control_reproduction.json', control)
                if not control['exact']:
                    raise RuntimeError('Original control did not reproduce exactly; treatment not trained')
            checkpoint = run / (condition + '_seed0.pt')
            torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()}, metadata=metadata(run,q,condition)), checkpoint)
            with torch.inference_mode():
                prediction = {k:v.cpu().numpy() for k,v in model(x,y['base']).items() if k!='actions'}
            artifact = run / (condition + '_predictions.npz')
            np.savez_compressed(artifact, **prediction)
            reports.append(dict(condition=condition, updates=1400, initial_state_sha256=q['initial_state_sha256'],
                checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), predictions=str(artifact),
                predictions_sha256=digest(artifact), history=history))
        verify(q['source_sha256'])
        write_json(run / 'training_result.json', dict(status='complete', reports=reports,
            elapsed_seconds=time.monotonic()-started, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            control_reproduced_exactly=True, interpretation='Training completion is not task-success improvement'))
        replace_json(run / 'training_progress.json', dict(phase='complete', models=2))
        print('S2L_TRAINING_COMPLETE', run, flush=True)
    except BaseException:
        write_json(run / 'training_failure.json', dict(completed_models=len(reports), traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)


def read_checkpoint(path, expected_condition, *, require_audit=True):
    path = Path(path).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if set(payload) != {'model','metadata'}:
        raise ValueError('Expected explicitly derived model and metadata')
    meta = payload['metadata']; run = output_path(Path(meta['training_run']))
    p, q, _, _, _ = load(run)
    result = read(run / 'training_result.json')
    if (meta['scope'] != SCOPE or meta['condition'] != expected_condition or expected_condition not in CONDITIONS
            or path != run / (expected_condition+'_seed0.pt') or meta != metadata(run,q,expected_condition)
            or result['status']!='complete' or len(result['reports'])!=2
            or {r['condition'] for r in result['reports']} != set(CONDITIONS)):
        raise ValueError('Incomplete or unregistered final paired checkpoint')
    for record in result['reports']:
        final = run/(record['condition']+'_seed0.pt')
        if (Path(record['checkpoint']).resolve()!=final or digest(final)!=record['checkpoint_sha256']
                or record['updates']!=1400 or record['initial_state_sha256']!=q['initial_state_sha256']):
            raise ValueError('Paired final checkpoint identity changed')
    control = compare_control(torch.load(run/(CONDITIONS[0]+'_seed0.pt'),map_location='cpu',weights_only=True)['model'])
    if not control['exact'] or not read(run/'control_reproduction.json')['exact']:
        raise ValueError('Original control reproduction failed')
    if require_audit:
        audit = read(run/'training_audit.json')
        if audit['passed'] is not True or audit['source_sha256'][str(run/'training_result.json')]!=digest(run/'training_result.json'):
            raise ValueError('Independent CPU training audit missing or changed')
    model = StructuredActionAdapter(q['channel_dim'],q['geometry_mean'])
    model.load_state_dict(payload['model'],strict=True)
    return payload


def audit(run):
    p,q,raw,labels,weights=load(run);x,y=tensors(raw),tensors(labels)
    xd=tensors(arrays(BASE_RUN/'development_inputs.npz'));yd=tensors(arrays(BASE_RUN/'development_labels.npz'))
    reports=[]
    for condition in CONDITIONS:
        payload=read_checkpoint(run/(condition+'_seed0.pt'),condition,require_audit=False)
        model=StructuredActionAdapter(q['channel_dim'],q['geometry_mean']);model.load_state_dict(payload['model']);model.eval()
        with torch.inference_mode():
            prediction=model(x,y['base']);dev=model(xd,yd['base'])
            assert torch.equal(model(x,y['base'],enabled=False)['actions'],y['base'])
        saved=arrays(run/(condition+'_predictions.npz'))
        errors={k:float(np.abs(prediction[k].numpy()-v).max()) for k,v in saved.items()}
        assert max(errors.values())<2e-5 and all(np.isfinite(v).all() for v in saved.values())
        assert torch.all(prediction['delta'].abs()<=model.caps+1e-6)
        groups=[]
        for task in TASKS:
            for kind in ('expert','anchor','recovery'):
                ids=[i for i,r in enumerate(p['rows']) if r['task']==task and r['kind']==kind]
                if not ids:continue
                w=y['row_weights'][ids];w=w/w.sum()
                error=float(action_direction_loss(prediction['delta'][ids],y['base'][ids],y['target'][ids],y['valid'][ids],w,model.caps))
                groups.append(dict(task=task,kind=kind,rows=len(ids),common_original_weight_direction_loss=error))
        dev_rms=(dev['delta'].square().mean(-1)*yd['valid']).sum(-1)/yd['valid'].sum(-1).clamp_min(1)
        reports.append(dict(condition=condition,cpu_reload_max_error=max(errors.values()),by_task_kind=groups,
            common_original_weight_train_loss=float(loss(model,x,y,y['row_weights'])),
            development_cup_anchor_native_rms=float(dev_rms.mean().sqrt()),
            development_cup_anchor_max_abs=float(dev['delta'].abs().max())))
    verify(q['source_sha256'])
    write_json(run/'training_audit.json',dict(passed=True,reports=reports,control_exact=True,
        source_sha256={str(run/'training_result.json'):digest(run/'training_result.json')},
        interpretation='All scores are fixed-cache diagnostics; held-out cup rows do not choose checkpoint/strength. Full-rollout efficacy remains untested.'))
    print(json.dumps(dict(passed=True,reports=reports),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('prepare','preflight','train','audit'));parser.add_argument('--run',type=Path)
    args=parser.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.phase=='prepare':
        if args.run is not None:parser.error('prepare creates a fresh isolated directory')
        prepare()
    else:
        if args.run is None:parser.error('--run required')
        {'preflight':preflight,'train':train,'audit':audit}[args.phase](output_path(args.run))


if __name__=='__main__':main()
