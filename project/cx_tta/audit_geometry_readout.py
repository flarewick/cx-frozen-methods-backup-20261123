"""Native point-API parity, target isolation, and CPU replay for S2h geometry."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import h5py
import torch
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read, verify
from .geometry_readout import GeometryReadout, geometry_loss, load, metrics, task_mean
from .geometry_labels import fixed_link_transform, ROLE_ACTORS
from .train_rotation_curriculum import state_hash


def native_point_api(runtime, articulation):
    """Execute only the trusted source's pure get_point method; never import simulator."""
    source = runtime / 'envs/utils/actor_utils.py'
    tree = ast.parse(source.read_text()); name = 'ArticulationActor' if articulation else 'Actor'
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_point')
    for a in fn.args.args: a.annotation = None
    fn.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])); scope = {'np': np}
    exec(compile(module, str(source), 'exec'), scope)
    return scope['get_point']


def label_audit(run):
    p, _, x, y = load(run, learning=False); rows = p['rows']; maximum = 0.; count = 0
    assert len(rows) == 592 and len(p['expert_indices']) == 572 and len(p['recovery_indices']) == 20
    for dd in p['expert_definitions']:
        ids = [i for i, r in enumerate(rows) if r['kind'] == 'expert' and r['scene_group'] == dd['group']]
        with h5py.File(rows[ids[0]]['source'], 'r') as h:
            frames = [rows[i]['frame'] for i in ids]
            state = h['joint_action/vector'][:][frames].astype(np.float32)
            np.testing.assert_array_equal(state, y['command_state'][ids])
            for slot, actor in enumerate(ROLE_ACTORS[dd['task']]):
                if actor:
                    raw = h['cx_privileged/' + actor][:][frames]
                    expected = np.broadcast_to(np.eye(4), (len(ids), 4, 4)).copy()
                    expected[:, :3, 3] = raw[:, :3]
                    expected[:, :3, :3] = Rotation.from_quat(raw[:, [4, 5, 6, 3]]).as_matrix()
                    np.testing.assert_allclose(expected, y['root_world'][ids, slot], atol=1e-12, rtol=0)
        for slot, spec in enumerate(dd['actors']):
            if spec is None:
                assert not y['root_valid'][ids, slot].any() and not y['functional_valid'][ids, slot].any(); continue
            articulation = dd['task'] == 'turn_switch'; api = native_point_api(Path(dd['runtime']), articulation)
            points = {'contact': 'contact_points', 'functional': 'functional_points'} if articulation else {'contact': 'contact_points_pose', 'functional': 'functional_matrix'}
            for i in ids:
                root = y['root_world'][i, slot]
                obj = SimpleNamespace(config=spec['config'], POINTS=points,
                    actor=SimpleNamespace(get_pose=lambda matrix=root: SimpleNamespace(to_transformation_matrix=lambda: matrix)))
                if articulation:
                    obj.link_dict = {}
                    for point in spec['config']['contact_points']:
                        t = fixed_link_transform(Path(spec['asset']) / 'mobility.urdf', point['base'], spec['config']['scale'])
                        if t is not None:
                            matrix = root @ t
                            obj.link_dict[point['base']] = SimpleNamespace(get_pose=lambda matrix=matrix: SimpleNamespace(to_transformation_matrix=lambda: matrix))
                for index in np.flatnonzero(y['contact_valid'][i, slot]):
                    expected = api(obj, 'contact', int(index), 'matrix'); actual = y['contact_world'][i, slot, index]
                    error = float(np.max(np.abs(actual - expected))); assert error < 1e-12; maximum = max(maximum, error); count += 1
                if y['functional_valid'][i, slot]:
                    kind = 'contact' if articulation else 'functional'
                    index = 1 if spec['actor'] == 'block' else 0
                    expected = api(obj, kind, index, 'matrix'); actual = y['functional_world'][i, slot]
                    error = float(np.max(np.abs(actual - expected))); assert error < 1e-12; maximum = max(maximum, error); count += 1
    for name in ('root', 'functional'):
        relative = y[name + '_relative_command_m']; ee = y['command_ee_world']
        recovered = np.einsum('naoi,naji->naoj', relative, ee[..., :3, :3]) + ee[:, :, None, :3, 3]
        np.testing.assert_allclose(recovered, np.broadcast_to(y[name + '_world'][:, None, :, :3, 3], recovered.shape), atol=1e-12, rtol=0)
    rel = y['functional_world'][:, 1, :3, 3] - y['functional_world'][:, 0, :3, 3]
    np.testing.assert_array_equal(rel[:572], y['relation_world_m'][:572])
    valid = y['relation_valid'][:572]
    assert valid.any() and not y['relation_valid'][572:].any() and not y['articulation_valid'].any()
    assert not y['functional_orientation_valid'][572:].any() and not y['contact_orientation_valid'][572:].any()
    for key in ('past_visual', 'past_text', 'past_state', 'past_ee_pose', 'has_history'):
        assert not x[key][:572].any()
    assert x['has_history'][572:].all() and all(np.isfinite(a).all() for a in y.values())
    expected = y['command_state'].copy(); expected[:, [0,1,2,3,4,5,7,8,9,10,11,12]] /= np.pi
    np.testing.assert_array_equal(expected, x['state'])
    with np.load(Path(p['source']) / 'probe_labels.npz') as z:
        np.testing.assert_array_equal(y['functional_world'][572:, 0, :3, 3], z['contacts_world'])
        np.testing.assert_allclose(y['command_ee_world'][572:, 1], z['ee'], atol=0, rtol=0)
    verify(p)
    write_json(run / 'label_audit.json', dict(passed=True, rows=592, expert_scenes=32, recovery_scenes=3,
        native_point_api_comparisons=count, native_api_max_error=maximum, relative_roundtrip_passed=True,
        masked_missing_articulation_and_orientation=True, no_invented_expert_history=True, observable_state_contract=True,
        asset_version_scope='Current read-only asset metadata frozen by SHA; no historical asset snapshot claim. Recovery nominal point independently matches fixed-link asset geometry within.1mm.',
        scope='Pure historical get_point code and stored observations, not simulation replay or success validation.',
        source_sha256={str(run / 'protocol.json'): digest(run / 'protocol.json'), str(Path(__file__)): digest(Path(__file__))}))
    print('GEOMETRY_LABEL_AUDIT_PASSED', count, maximum, flush=True)


def preflight(run):
    p, q, raw, labels = load(run); x = {k: torch.from_numpy(v) for k, v in raw.items()}
    y = torch.from_numpy(labels['regression']); valid = torch.from_numpy(labels['regression_valid']); checks = []
    for job in q['jobs']:
        weights = torch.tensor(job['weights'], dtype=torch.float32); held = job['held']
        selected = np.flatnonzero(np.array(job['weights']) > 0)
        if job['condition'] != 'expert_plus_include':
            assert set(selected).isdisjoint(held)
            assert not {p['rows'][i]['scene_group'] for i in selected} & {job['scene']}
        for seed in q['seeds']:
            init = next(v for v in q['initializations'] if v['scene'] == job['scene'] and v['seed'] == seed)
            model = GeometryReadout(p['channel_dim'], job['mean'])
            model.load_state_dict(torch.load(init['path'], map_location='cpu', weights_only=True))
            assert state_hash(model) == init['state_sha256'] and sum(v.numel() for v in model.parameters()) == q['parameters']
            with torch.no_grad():
                out = model(x)
                np.testing.assert_allclose(out['regression'].numpy(), np.tile(job['mean'], (len(y), 1)), atol=5e-7, rtol=0)
        torch.manual_seed(8301); model = GeometryReadout(p['channel_dim'], job['mean'])
        with torch.no_grad(): model.head.weight.normal_(0, .03); model.query.weight.normal_(0, .01)
        assert not any(isinstance(v, (torch.nn.BatchNorm1d, torch.nn.Dropout)) for v in model.modules())
        def calc(data, targets=y, mask=valid):
            loss = geometry_loss(model(data)['regression'], targets, mask, weights)
            gradient = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
            return float(loss.detach()), [None if v is None else v.detach() for v in gradient]
        def difference(a, b):
            assert all((v is None) == (w is None) for v, w in zip(a[1], b[1]))
            return max([abs(a[0] - b[0])] + [float((v-w).abs().max()) for v,w in zip(a[1],b[1]) if v is not None])
        original = calc(x); zero = np.flatnonzero(np.array(job['weights']) == 0)
        poisoned = {k: v.clone() for k,v in x.items()}
        for v in poisoned.values(): v[zero] = 19.25
        yy = y.clone(); yy[zero] = 123.; vv = valid.clone(); vv[zero] = True
        isolated = difference(original, calc(poisoned, yy, vv)); assert isolated < 1e-6
        yy = y.clone(); yy[~valid] = 999.; masked = difference(original, calc(x, yy)); assert masked < 1e-6
        yy = y.clone(); yy[int(selected[0]), 0] += .1
        sensitivity = difference(original, calc(x, yy)); assert sensitivity > 1e-6
        if job['condition'] == 'expert_plus_include':
            yy = y.clone(); yy[held[0], 0] += .1
            assert difference(original, calc(x, yy)) > 1e-6
        checks.append(dict(scene=job['scene'], condition=job['condition'], zero_weight_input_label_gradient_error=isolated,
                           masked_target_gradient_error=masked, active_target_gradient_change=sensitivity))
    for scene in {j['scene'] for j in q['jobs']}:
        jobs = [j for j in q['jobs'] if j['scene'] == scene]
        assert all(j['mean'] == jobs[0]['mean'] for j in jobs)
        ew = np.array(q['expert_weights']); rw = np.array(q['recovery_weights'])
        for j in jobs:
            expected = rw.copy()
            if j['condition'] != 'recovery_only_omit': expected += ew
            if j['condition'] != 'expert_plus_include': expected[j['held']] = 0
            np.testing.assert_array_equal(expected, j['weights'])
    verify(p); verify(q)
    write_json(run / 'learning_preflight.json', dict(passed=True, checks=checks, models=27,
        paired_initializations=True, expert_and_recovery_weights_frozen=True, held_mean_and_pca_isolated=True,
        source_sha256={str(run / 'learning_protocol.json'): digest(run / 'learning_protocol.json'), str(Path(__file__)): digest(Path(__file__))}))
    print('GEOMETRY_PREFLIGHT_PASSED', len(checks), flush=True)


def finalize(run):
    p, q, raw, labels = load(run); result = read(run / 'learning_result.json')
    assert result['status'] == 'complete' and len(result['reports']) == 27
    x = {k: torch.from_numpy(v) for k,v in raw.items()}; reports = []; seen = set(); error = 0.; baselines = []
    for record in result['reports']:
        key = (record['scene'], record['condition'], record['seed']); assert key not in seen; seen.add(key)
        job = next(j for j in q['jobs'] if (j['scene'], j['condition']) == key[:2])
        init = next(v for v in q['initializations'] if (v['scene'], v['seed']) == (key[0], key[2]))
        assert init['state_sha256'] == record['initial_sha256'] and record['updates'] == q['updates']
        for a,b in (('checkpoint','checkpoint_sha256'),('artifact','sha256')): assert digest(Path(record[a])) == record[b]
        model = GeometryReadout(p['channel_dim'], job['mean']); model.load_state_dict(torch.load(record['checkpoint'], map_location='cpu', weights_only=True)); model.eval()
        with torch.no_grad(): again = {k: v.numpy() for k,v in model(x).items()}
        with np.load(record['artifact']) as z: stored = {k:z[k].copy() for k in z.files}
        assert set(stored) == set(again) and all(np.isfinite(v).all() for v in stored.values())
        diff = max(float(np.abs(again[k]-stored[k]).max()) for k in again); assert diff < 2e-5; error = max(error, diff)
        for k in ('current_attention','past_attention'):
            assert (stored[k] >= 0).all(); np.testing.assert_allclose(stored[k].sum(-1), 4, atol=2e-6)
        report = dict(scene=key[0], condition=key[1], seed=key[2])
        report['held'] = metrics(stored['regression'], labels, job['held'], job['held_weights'])
        report['train'] = metrics(stored['regression'], labels, job['train'], np.array(job['weights'])[job['train']])
        report['expert_by_task'] = {}
        for task in sorted({r['task'] for r in p['rows'] if r['kind']=='expert'}):
            ids = [i for i in p['expert_indices'] if p['rows'][i]['task'] == task]
            report['expert_by_task'][task] = metrics(stored['regression'], labels, ids, np.array(q['expert_weights'])[ids])
        # Rigid transforms preserve position errors; recompute held wrist-relative functional error independently.
        ids = job['held']; ee = labels['command_ee_world'][ids,1]; point = stored['regression'][ids,6:9] * .1
        relative = np.einsum('ni,nij->nj', point-ee[:,:3,3], ee[:,:3,:3])
        truth = labels['functional_relative_command_m'][ids,1,0]
        cm = float(np.array(job['held_weights']) @ np.linalg.norm(relative-truth,axis=1)*100)
        assert abs(cm-report['held']['primary']['functional_position_cm']) < 2e-5
        report['held_right_command_relative_functional_cm'] = cm
        reports.append(report)
    assert seen == {(j['scene'],j['condition'],s) for j in q['jobs'] for s in q['seeds']}
    signals = []
    for job in [j for j in q['jobs'] if j['condition']=='expert_plus_omit']:
        prior = task_mean(np.tile(job['mean'],(len(p['rows']),1)),labels,p['rows'],job['weights'],job['held'])
        baseline = metrics(prior,labels,job['held'],job['held_weights']); baselines.append(dict(scene=job['scene'],method='training_task_mean',held=baseline))
        pairs=[]
        for seed in q['seeds']:
            a=next(r for r in reports if (r['scene'],r['condition'],r['seed'])==(job['scene'],'expert_plus_omit',seed))['held']['primary']
            b=next(r for r in reports if (r['scene'],r['condition'],r['seed'])==(job['scene'],'recovery_only_omit',seed))['held']['primary']
            base=baseline['primary']
            pairs.append(all(a[k]<b[k] and a[k]<base[k] for k in ('root_position_cm','functional_position_cm')) and a['face_angle_deg']<=b['face_angle_deg']+1)
        signals.append(dict(scene=job['scene'],paired_seed_pass=pairs,consistent_improvement=all(pairs)))
    verify(p); verify(q)
    write_json(run/'learning_audit.json',dict(passed=True,models=27,cpu_prediction_max_error=error,relative_position_metrics_parity=True,
        source_sha256={str(run/'learning_result.json'):digest(run/'learning_result.json'),str(Path(__file__)):digest(Path(__file__))}))
    write_json(run/'summary.json',dict(status='complete',reports=reports,baselines=baselines,scene_signals=signals,
        development_signal=sum(v['consistent_improvement'] for v in signals)>=2,
        scope='Only3reused developer switch recovery scenes; include is fit, experts are train. No action/task-success or cross-task generalization result.',
        source_sha256={str(run/'learning_audit.json'):digest(run/'learning_audit.json'),str(run/'learning_protocol.json'):digest(run/'learning_protocol.json'),str(run/'learning_result.json'):digest(run/'learning_result.json')}))
    lines=['# S2h: final600, all three paired seeds','','All geometry metrics are offline. Include condition trains on the evaluated scene.','',
           '|Scene|Condition|Root cm mean (min–max)|Functional cm mean (min–max)|Root rotation °|Face °|Geometry gates|',
           '|---|---|---|---|---|---|---|']
    for scene in sorted({r['scene'] for r in reports}):
        for condition in q['conditions']:
            values=[r['held']['primary'] for r in reports if r['scene']==scene and r['condition']==condition]
            root=[v['root_position_cm'] for v in values];fun=[v['functional_position_cm'] for v in values]
            lines.append(f'|{scene}|{condition}|{np.mean(root):.2f} ({min(root):.2f}–{max(root):.2f})|{np.mean(fun):.2f} ({min(fun):.2f}–{max(fun):.2f})|{np.mean([v["root_angle_deg"] for v in values]):.2f}|{np.mean([v["face_angle_deg"] for v in values]):.2f}|{sum(v["geometry_gate"] for v in values)}/3|')
    lines += ['',f'Prespecified development signal: {sum(v["consistent_improvement"] for v in signals)>=2}.',
              'Training-task-mean is an evaluator-metadata diagnostic prior; no task ID is forwarded to the model.']
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
    print('GEOMETRY_FINAL_AUDIT_PASSED',error,signals,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--phase',choices=('labels','preflight','final'),default='final');a=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2)
    {'labels':label_audit,'preflight':preflight,'final':finalize}[a.phase](a.run.resolve())
