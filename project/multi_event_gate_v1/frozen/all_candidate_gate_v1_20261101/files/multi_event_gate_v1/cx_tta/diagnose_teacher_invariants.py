"""S2f supplement: contact invariants, full-background grip coverage and probe fit control."""
from pathlib import Path
from itertools import combinations
import numpy as np
from threadpoolctl import threadpool_limits
from .paths import ROOT,write_json
from .audit import digest
from .probe_candidates import read,verify
from .diagnose_teacher_observability import PAIRED,CACHE,transform_blocks,balanced

RUN=ROOT/'outputs/reports/diagnostics/teacher_observability_20261003_044742_113205'


def interpolate(x,y,train,weights):
    mx=weights@x[train];my=weights@y[train];sw=np.sqrt(weights)
    design=(x[train]-mx)*sw[:,None]
    inverse=np.linalg.pinv(design@design.T,rcond=1e-10)
    return (x-mx)@design.T@inverse@((y[train]-my)*sw[:,None])+my


def main():
    first=read(RUN/'result.json');p=read(PAIRED/'protocol.json');cp=read(CACHE/'protocol.json')
    official=Path(next(b['run'] for b in first['branches'] if b['point']==4))
    source=Path(read(official/'manifest.json')['source'])/'robotwin/envs'
    code=[source/'_base_task.py',source/'robot/robot.py',source/'_GLOBAL_CONFIGS.py',source/'utils/actor_utils.py']
    write_json(RUN/'supplement_protocol.json',dict(status='frozen_before_supplement',
        reason='Initial alpha1 diagnostic is strongly shrunk; add a fixed minimum-norm interpolation fit control and geometry1NN. '
            'Report all results, no held-label hyperparameter selection or adapter update.',
        endpoint_contract='Official get_grasp_pose uses -0.12-pre_grasp_dis along tool x; switch pre_grasp_dis0.04. '
            'Thus planned wrist endpoint plus0.16m x at approach /0.12m x at contact reconstructs functional contact point. '
            'Task-specific constants are diagnostic only, not proposed shared controller inputs.',
        probes='Same four legal feature families, same scene/branch/row weights and train-only block RMS. '
            'Minimum-norm dual pseudoinverse rcond1e-10, nearest neighbor, and training-only mean. No grid search. '
            'Predict invariant contact point, never teacher-selected approach pose. Phase only a diagnostic label.',
        source_sha256={str(f):digest(f) for f in [Path(__file__),ROOT/'cx_tta/diagnose_teacher_observability.py',RUN/'result.json',
            PAIRED/'protocol.json',PAIRED/'data.npz',CACHE/'protocol.json',CACHE/'features.npz',CACHE/'data.npz',*code]}))
    points=[];invariant=[]
    for b in first['branches']:
        if b['task']!='turn_switch':continue
        trace=[__import__('json').loads(x) for x in (Path(b['run'])/'trace.jsonl').read_text().splitlines()]
        fork=read(Path(b['run'])/'fork.json')
        for plan,distance in ((1,.16),(2,.12)):
            record=next((v for v in b['plans'] if v['plan']==plan),None)
            if record is None:continue
            pose=np.array(record['endpoint_poses'][1]);point=pose[:3,3]+distance*pose[:3,0]
            points.append(dict(seed=b['seed'],point=b['point'],source=b['run'],qualified=b['qualified'],plan=plan,
                contact_xyz=point.tolist(),root_xyz=b['switch_root_pose'][:3]))
        invariant.append(dict(point=b['point'],scene=b['seed'],qualified=b['qualified'],
            switch_q_fork=fork['diagnostics']['physical']['offline_only']['articulation_qpos']['switch'][0],
            switch_q_after_closure=trace[19]['offline_only']['physical']['offline_only']['articulation_qpos']['switch'][0]))
    spreads=[]
    for seed in (421000,421001,421007):
        for plan in (1,2):
            q=[v for v in points if v['seed']==seed and v['plan']==plan]
            maximum=max(np.linalg.norm(np.array(a['contact_xyz'])-b['contact_xyz']) for a,b in combinations(q,2))
            assert maximum<1e-4
            spreads.append(dict(scene=seed,plan=plan,branches=len(q),max_contact_spread_m=float(maximum)))
    with np.load(PAIRED/'data.npz') as z:context=z['context'].copy();target=z['target'].copy();valid=z['valid'].copy()
    with np.load(p['preprocess_source']) as z:states=(context*z['context_std']+z['context_mean'])[:,64:78]
    coverage=[]
    for kind in ('expert','anchor','recovery'):
        for task in sorted({r['task'] for r in p['rows']}):
            ids=[i for i in p['train_indices'] if p['rows'][i]['kind']==kind and p['rows'][i]['task']==task and p['rows'][i]['group']!='turn_switch:421007']
            hits=[i for i in ids if states[i,13]>.5 and np.any(target[i,:,13][valid[i]]<.1)]
            coverage.append(dict(kind=kind,task=task,rows=len(ids),open_to_closed_rows=len(hits),
                matching_groups=sorted({p['rows'][i]['group'] for i in hits})))
    which=np.array([i for i,r in enumerate(cp['rows']) if r['task']=='turn_switch']);rows=[cp['rows'][i] for i in which]
    fi=np.array(cp['feature_current_indices'])[which];hi=np.array(cp['feature_history_indices'])[fi]
    with np.load(CACHE/'features.npz') as z:joint=z['joint'].astype(float);state=z['state'].astype(float)
    state[:,[0,1,2,3,4,5,7,8,9,10,11,12]]/=np.pi
    with np.load(CACHE/'data.npz') as z:coarse=z['coarse'][which].copy();phase=z['plan_ids'][which,0].copy()
    semantic=context[np.array(p['new_indices'])[which]]
    blocks={'proprio_history':[state[fi],state[hi]],
        'semantic_pca_history':[semantic[:,:64],semantic[:,64:78],semantic[:,78:142],semantic[:,142:156]],
        'raw_joint_history':[joint[fi],state[fi],joint[hi],state[hi]],
        'coarse_history':[coarse[:,:64],coarse[:,64:78],coarse[:,78:142],coarse[:,142:156]]}
    contact=np.array([next(v['contact_xyz'] for v in points if v['source']==r['source'] and v['plan']==1) for r in rows]);onehot=np.eye(4)[phase]
    results=[]
    for scene in sorted({r['scene_group'] for r in rows}):
        train=np.array([i for i,r in enumerate(rows) if r['scene_group']!=scene]);held=np.array([i for i,r in enumerate(rows) if r['scene_group']==scene])
        tw=balanced(rows,train);hw=balanced(rows,held)
        for name,bs in blocks.items():
            x=transform_blocks(bs,train);pred=interpolate(x,contact,train,tw);scores=interpolate(x,onehot,train,tw)
            altered=contact.copy();altered[held]=999
            assert np.array_equal(pred,interpolate(x,altered,train,tw))
            d=((x[held,None]-x[train][None])**2).sum(-1);nearest=train[np.argmin(d,axis=1)]
            prior=tw@contact[train]
            rank=int(np.linalg.matrix_rank(x[train]-(tw@x[train])))
            results.append(dict(scene=scene,representation=name,train_rows=len(train),centered_rank=rank,
                contact_train_error_m=float(tw@np.linalg.norm(pred[train]-contact[train],axis=1)),
                contact_held_error_m=float(hw@np.linalg.norm(pred[held]-contact[held],axis=1)),
                contact_nn_error_m=float(hw@np.linalg.norm(contact[nearest]-contact[held],axis=1)),
                contact_prior_error_m=float(hw@np.linalg.norm(prior-contact[held],axis=1)),
                phase_train_accuracy=float(tw@(np.argmax(scores[train],axis=1)==phase[train])),
                phase_held_accuracy=float(hw@(np.argmax(scores[held],axis=1)==phase[held])),
                geometry_target_poisoning_passed=True))
    verify(read(RUN/'supplement_protocol.json'))
    write_json(RUN/'invariants.json',dict(status='complete',functional_points=points,contact_spreads=spreads,
        articulation_drift=invariant,background_open_close_coverage=coverage,minimum_norm_probes=results,
        interpretation='Same-scene contact invariant holds, although wrist target choice can differ by grid angle. '
            'This is geometric consistency, not proof of collision-free feasibility or successful alternative action. '
            'No absent-evidence claim that RGB cannot identify geometry;20rows/3appearance+instruction+geometry bundles are insufficient to identify causal source of failure.'))
    print('INVARIANTS',spreads,flush=True)
    print('PROBES',results,flush=True)

if __name__=='__main__':
    threadpool_limits(2);main()
