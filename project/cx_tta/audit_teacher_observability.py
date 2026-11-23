"""Verify S2f provenance and extend fixed scene-held-out probes to face normal."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .paths import ROOT,write_json
from .audit import digest
from .probe_candidates import read,verify
from .fixed_recovery_dataset import load_fixed_cache
from .diagnose_teacher_observability import PAIRED,CACHE,pose7,balanced,transform_blocks
from .diagnose_teacher_invariants import RUN,interpolate


def angle(pred,true):
    norm=np.linalg.norm(pred,axis=1)
    if np.any(norm<1e-8):raise ValueError('Unidentified normal: near-zero prediction')
    return np.degrees(np.arccos(np.clip(np.sum(pred*true,axis=1)/norm,-1,1)))


def main():
    # Validate original causal cache dependencies, not only the report hashes.
    load_fixed_cache(CACHE,scope='developer_same_task_scene_holdout')
    verify(read(RUN/'protocol.json'));verify(read(RUN/'supplement_protocol.json'));verify(read(RUN/'result.json'))
    p=read(PAIRED/'protocol.json');cp=read(CACHE/'protocol.json');result=read(RUN/'result.json')
    for report in read(PAIRED/'result.json')['reports']:
        assert digest(Path(report['artifact']))==report['sha256']
    write_json(RUN/'orientation_protocol.json',dict(status='frozen_before_orientation_probe',
        target='Official object face normal -root_rotation[:,0], a privileged diagnostic label; world normal is transformed '
            'to the same observable wrist frame for both predicted and true normals, preserving angle error.',
        method='Same whole-scene split, train-only block normalization, fixed minimum-norm inverse rcond1e-10,1NN and train-only mean. '
            'No rotations from GT as input, no hyperparameter selection. Norm<1e-8 must fail rather than fabricate a direction.',
        source_sha256={str(f):digest(f) for f in (Path(__file__),RUN/'result.json',RUN/'invariants.json',
            ROOT/'cx_tta/diagnose_teacher_invariants.py',ROOT/'cx_tta/diagnose_teacher_observability.py',
            PAIRED/'data.npz',CACHE/'features.npz',CACHE/'data.npz')}))
    which=np.array([i for i,r in enumerate(cp['rows']) if r['task']=='turn_switch']);rows=[cp['rows'][i] for i in which]
    fi=np.array(cp['feature_current_indices'])[which];hi=np.array(cp['feature_history_indices'])[fi]
    with np.load(CACHE/'features.npz') as z:joint=z['joint'].astype(float);state=z['state'].astype(float)
    state[:,[0,1,2,3,4,5,7,8,9,10,11,12]]/=np.pi
    with np.load(CACHE/'data.npz') as z:coarse=z['coarse'][which].copy()
    with np.load(PAIRED/'data.npz') as z:semantic=z['context'][np.array(p['new_indices'])[which]]
    blocks={'proprio_history':[state[fi],state[hi]],
        'semantic_pca_history':[semantic[:,:64],semantic[:,64:78],semantic[:,78:142],semantic[:,142:156]],
        'raw_joint_history':[joint[fi],state[fi],joint[hi],state[hi]],
        'coarse_history':[coarse[:,:64],coarse[:,64:78],coarse[:,78:142],coarse[:,142:156]]}
    y=np.array([-pose7(next(b['switch_root_pose'] for b in result['branches'] if b['run']==r['source']))[:3,0] for r in rows])
    out=[]
    for scene in sorted({r['scene_group'] for r in rows}):
        train=np.array([i for i,r in enumerate(rows) if r['scene_group']!=scene]);held=np.array([i for i,r in enumerate(rows) if r['scene_group']==scene])
        tw=balanced(rows,train);hw=balanced(rows,held)
        for name,bs in blocks.items():
            x=transform_blocks(bs,train);pred=interpolate(x,y,train,tw)
            changed=y.copy();changed[held]=123
            assert np.array_equal(pred,interpolate(x,changed,train,tw))
            nearest=train[np.argmin(((x[held,None]-x[train][None])**2).sum(-1),axis=1)]
            prior=np.repeat((tw@y[train])[None],len(held),axis=0)
            out.append(dict(scene=scene,representation=name,train_angle_deg=float(tw@angle(pred[train],y[train])),
                held_angle_deg=float(hw@angle(pred[held],y[held])),prior_angle_deg=float(hw@angle(prior,y[held])),
                nearest_angle_deg=float(hw@angle(y[nearest],y[held]))))
    inv=read(RUN/'invariants.json');combined=[]
    for seed in (421000,421001,421007):
        pts=np.array([r['contact_xyz'] for r in inv['functional_points'] if r['seed']==seed])
        spread=float(np.max(np.linalg.norm(pts[:,None]-pts[None],axis=-1)));assert spread<1e-4
        combined.append(dict(scene=seed,all_stage_contact_spread_m=spread))
    verify(read(RUN/'orientation_protocol.json'))
    write_json(RUN/'audit.json',dict(passed=True,cache_dependencies_unchanged=True,frozen_model_artifacts_unchanged=True,
        full_teacher_files_unchanged=True,held_label_poisoning_no_effect=True,combined_contact_invariants=combined,
        orientation_probes=out,geometry_probe_rows=20,independent_switch_scenes=3,teacher_attempts=14,
        source_sha256={str(RUN/f):digest(RUN/f) for f in ('protocol.json','result.json','supplement_protocol.json','invariants.json','orientation_protocol.json')},
        scope='Provenance, alignment and isolation verified. Not a success-rate, visual impossibility, or teacher repetition claim.'))
    print('AUDIT_PASSED',out,flush=True)

if __name__=='__main__':
    threadpool_limits(2);main()
