"""S2m fixed-cache expert-sequence history contrast; original S0 weights.

Saved expert frames are causal history, not certified native VLA queries.
All targets/current inputs and original recovery history remain immutable.
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
from .train_task_balanced_action import arrays, weight_summary, state_hash_from_state

SCOPE = 'S2m fixed-cache causal expert-sequence history residual'
CONDITIONS = ('S0_control', 'S_expert_history')
TASKS = ('beat_block_hammer', 'place_shoe', 'turn_switch', 'place_empty_cup')
BASE_RUN = ROOT / 'outputs/effect_learning/structured_action_20261003_095511_385074'
DESIGN = ROOT / 'outputs/reports/attribution/task_balanced_accepted_20261004_135836_691479/next_experiments.json'
DESIGN_SHA = 'a2c863d78bc765ac64f929beb709d55841069199e8a0d278361e7e485196bb9b'
PAST_FIELDS = ('visual', 'text', 'state', 'ee_pose')


def expert_mapping(rows):
    lookup = {}
    for i, row in enumerate(rows):
        if row['training_row'] != i:
            raise ValueError('Training row order changed')
        scene = int(row['group'].rsplit(':', 1)[1])
        if 100380 <= scene <= 100399:
            raise ValueError('Development scene leaked into training')
        if row['kind'] != 'expert':
            continue
        source = row['source_query']
        if source['kind'] != 'expert_hdf5' or source['frame'] != row['frame']:
            raise ValueError('Explicit saved expert frame identity required')
        key = (row['task'], row['group'], source['path'], row['instruction'], row['frame'])
        if key in lookup:
            raise ValueError('Duplicate expert source/frame identity')
        lookup[key] = i
    result = []
    for key, current in lookup.items():
        frame = key[-1]
        past = lookup.get((*key[:-1], frame - 32))
        if past is not None:
            if frame < 32 or rows[past]['frame'] >= frame:
                raise ValueError('Future/self history forbidden')
            result.append(dict(training_row=current, past_training_row=past))
    return sorted(result, key=lambda pair: pair['training_row'])


def add_history(control, mapping):
    result = {key: value.copy() for key, value in control.items()}
    if set(result) != FORWARD_KEYS:
        raise ValueError('Exactly nine legal observable fields required')
    for pair in mapping:
        current, past = pair['training_row'], pair['past_training_row']
        if control['has_history'][current].item() != 0:
            raise ValueError('Expert history must originally be absent')
        for field in PAST_FIELDS:
            result['past_' + field][current] = control[field][past]
        result['has_history'][current] = 1
    return result


def exact_arrays(left, right):
    return (set(left) == set(right) and all(left[k].dtype == right[k].dtype
        and left[k].shape == right[k].shape and left[k].tobytes() == right[k].tobytes() for k in left))


def parent_data():
    if digest(DESIGN) != DESIGN_SHA:
        raise ValueError('Sealed S2m design changed')
    design = read(DESIGN)
    if design['stage'] != 'S2m' or design['status'] != 'design_frozen_before_cache_preparation_or_training':
        raise ValueError('Frozen unexecuted design required')
    verify(design['source_sha256'])
    accepted = read(Path(design['accepted_result']))
    p, q = read(BASE_RUN / 'protocol.json'), read(BASE_RUN / 'training_protocol.json')
    verify(q['source_sha256'])
    gaps = accepted['training_support']['inherited_missing_sources']
    missing = sorted(path for path in p['source_sha256'] if not Path(path).is_file())
    if len(p['source_sha256']) != 507 or missing != sorted(gaps['missing_sources']) or len(missing) != 33:
        raise ValueError('Inherited raw-source gaps changed')
    x, y = arrays(BASE_RUN / 'inputs.npz'), arrays(BASE_RUN / 'labels.npz')
    if len(p['rows']) != 611 or len(x['state']) != 611 or set(x) != FORWARD_KEYS:
        raise ValueError('Original fixed611 cache required')
    if (q['seed'], q['updates'], q['lr'], q['weight_decay'], q['channel_dim'], q['parameters']) != (0,1400,.0003,.01,32,42934):
        raise ValueError('Original training contract changed')
    mapping = expert_mapping(p['rows'])
    counts = Counter(p['rows'][r['training_row']]['task'] for r in mapping)
    if len(mapping) != 444 or dict(counts) != design['experiment']['eligible_by_task']:
        raise ValueError('Preregistered eligible history population changed')
    inventory = {}
    for source in accepted['training_support']['expert_hdf5_sources']:
        for pair in source['cached_saved_frame_pairs']:
            if pair['training_row'] in inventory:
                raise ValueError('Duplicate accepted mapping')
            inventory[pair['training_row']] = dict(pair, source_path=source['path'],
                source_sha256=source['sha256'], history_domain='expert_saved_sequence',
                certified_native_query_history=False)
    detailed = []
    for pair in mapping:
        record = inventory.pop(pair['training_row'])
        row = p['rows'][pair['training_row']]
        if (record['past_training_row'] != pair['past_training_row'] or record['saved_frame_gap'] != 32
                or record['source_path'] != row['source_query']['path'] or record['physics_elapsed_s'] <= 0
                or record['physics_tick_gap'] <= 0):
            raise ValueError('Accepted causal source/time identity differs')
        detailed.append(dict(record, task=row['task'], group=row['group'],
            current_saved_frame=row['frame'], past_saved_frame=p['rows'][pair['past_training_row']]['frame']))
    if inventory:
        raise ValueError('Extra accepted mappings')
    if sum(float(x['has_history'][r['training_row'],0]) for r in p['rows'] if r['kind']=='expert') != 0:
        raise ValueError('Original expert history changed')
    if int(x['has_history'].sum()) != 27 or len(p['development_rows']) != 4:
        raise ValueError('Original history/held-out support changed')
    return p, q, x, y, detailed, gaps


def prepare():
    p, q, x, y, mapping, gaps = parent_data()
    treatment = add_history(x, mapping)
    run = new_run('expert_history_action', group=('effect_learning',))
    for condition, values in zip(CONDITIONS, (x, treatment)):
        np.savez_compressed(run / (condition + '_inputs.npz'), **values)
    write_json(run / 'history_mapping.json', dict(domain='expert_saved_sequence',pairs=mapping,
        original_recovery_history_unchanged=True, actor_receives_source_or_time_metadata=False))
    sources = dict(q['source_sha256']); sources.update(read(DESIGN)['source_sha256'])
    paths = [DESIGN, BASE_RUN/'S0_predictions.npz', Path(p['config']), Path(p['projection']),
        run/'history_mapping.json', *(run/(c+'_inputs.npz') for c in CONDITIONS)]
    code = [Path(__file__), ROOT/'scripts/train_expert_history_action.sh', ROOT/'tests/test_expert_history_action.py',
        ROOT/'cx_tta/train_task_balanced_action.py']
    snapshot = run/'source_snapshot'; snapshot.mkdir()
    for path in code:
        (snapshot/path.name).write_bytes(path.read_bytes())
    for path in paths+code:
        sources[str(path)] = digest(path)
    original = torch.load(BASE_RUN/'S0_seed0.pt', map_location='cpu', weights_only=True)
    write_json(run/'training_protocol.json', dict(schema_version=1,scope=SCOPE,status='frozen_before_training',
        conditions=list(CONDITIONS), parent_run=str(BASE_RUN), design=str(DESIGN),design_sha256=DESIGN_SHA,
        data_mode='Explicit fixed-cache expert history derivation',raw_teacher_revalidation_claim=False,
        inherited_reference_count=507,inherited_missing_sources=gaps,source_sha256=sources,
        initial=q['initial'],initial_state_sha256=q['initial_state_sha256'],seed=0,updates=1400,
        lr=q['lr'],weight_decay=q['weight_decay'],channel_dim=q['channel_dim'],parameters=q['parameters'],
        geometry_mean=q['geometry_mean'],caps_native_units=q['caps_native_units'],
        original_checkpoint_sha256=digest(BASE_RUN/'S0_seed0.pt'),
        original_model_state_sha256=state_hash_from_state(original['model']),
        projection=p['projection'],robot_config=p['config'],
        inputs={c:str(run/(c+'_inputs.npz')) for c in CONDITIONS},labels=str(BASE_RUN/'labels.npz'),
        eligible_history_rows=444,history_domain='expert_saved_sequence',certified_native_query_history=False,
        weight_summaries={c:weight_summary(p['rows'],y['row_weights']) for c in CONDITIONS},
        change='Only444 expert past fields and has_history; original S0 weighting, targets and real recovery history fixed',
        action_loss=q['action_loss'],auxiliary_coefficient=0.,batch='All611 fixed-order rows every update',
        checkpoint='Final1400 seed0 only',control_reproduction='Exact original S0 tensor equality before treatment',
        time_limit_seconds=600,backbone_trainable_parameters=0))
    print('RUN',run,flush=True)


def load(run):
    q = read(run/'training_protocol.json')
    if q['scope'] != SCOPE or q['conditions'] != list(CONDITIONS) or q['seed'] != 0 or q['updates'] != 1400:
        raise ValueError('Invalid paired expert-history contract')
    verify(q['source_sha256'])
    p, old, x, y, mapping, gaps = parent_data()
    if read(run/'history_mapping.json')['pairs'] != mapping or q['inherited_missing_sources'] != gaps:
        raise ValueError('Frozen history mapping/gaps changed')
    raw = {c:arrays(run/(c+'_inputs.npz')) for c in CONDITIONS}
    if not exact_arrays(raw[CONDITIONS[0]],x) or not exact_arrays(raw[CONDITIONS[1]],add_history(x,mapping)):
        raise ValueError('Forbidden cache changes or incorrect expert past copying')
    return p,q,raw,y,{c:y['row_weights'] for c in CONDITIONS}


def loss(model, x, y, weights):
    output = model(x, y['base'])
    action = action_direction_loss(output['delta'],y['base'],y['target'],y['valid'],weights,model.caps)
    return action + 0. * action.new_zeros(())


def preflight(run):
    p,q,raw,labels,weights=load(run);y=tensors(labels);checks=[]
    for condition in CONDITIONS:
        x=tensors(raw[condition])
        model=StructuredActionAdapter(q['channel_dim'],q['geometry_mean'])
        model.load_state_dict(torch.load(q['initial'],map_location='cpu',weights_only=True))
        assert state_hash(model)==q['initial_state_sha256']
        with torch.no_grad():
            output=model(x,y['base'])
            assert torch.equal(output['actions'],y['base']) and not output['delta'].any()
        total=loss(model,x,y,y['row_weights']);total.backward()
        assert torch.isfinite(total) and model.action_head[-1].weight.grad.abs().max()>0
        assert all(param.grad is None for param in model.geometry_head.parameters())
        checks.append(dict(condition=condition,identity_actions=True,geometry_excluded=True,
            history_rows=int(raw[condition]['has_history'].sum()),finite_initial_loss=float(total.detach())))
    write_json(run/'training_preflight.json',dict(passed=True,checks=checks,GPU_used=False,
        original_labels_weights_current_fields_exact=True,unchanged_noneligible_rows=167,
        expert_history_rows=444,recovery_history_rows=27,held_out_cup_rows=4,
        inherited_source_gaps_recorded=True,source_sha256={str(run/'training_protocol.json'):digest(run/'training_protocol.json')}))
    print('S2M_PREFLIGHT_PASSED',flush=True)

def metadata(run, q, condition):
    return dict(scope=SCOPE, condition=condition, training_run=str(run), channels=q['channel_dim'],
        geometry_mean=q['geometry_mean'], caps_native_units=q['caps_native_units'], parameters=q['parameters'],
        projection_path=q['projection'], projection_sha256=digest(Path(q['projection'])),
        robot_config_path=q['robot_config'], robot_config_sha256=digest(Path(q['robot_config'])),
        updates=1400, seed=0, initial_state_sha256=q['initial_state_sha256'], auxiliary_coefficient=0.,
        training_protocol_sha256=digest(run / 'training_protocol.json'), input_cache_sha256=digest(run / (condition+'_inputs.npz')), history_mapping_sha256=digest(run/'history_mapping.json'),
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
    y = tensors(labels, 'cuda')
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
            x = tensors(raw[condition], 'cuda')
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
        print('S2M_TRAINING_COMPLETE', run, flush=True)
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
    p,q,raw,labels,weights=load(run);y=tensors(labels)
    xd=tensors(arrays(BASE_RUN/'development_inputs.npz'));yd=tensors(arrays(BASE_RUN/'development_labels.npz'))
    reports=[]
    for condition in CONDITIONS:
        x=tensors(raw[condition])
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
