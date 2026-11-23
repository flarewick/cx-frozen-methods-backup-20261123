"""S2g observable interface using expert-only shared channel PCA, no policy updates."""
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from .paths import ROOT,new_run,write_json
from .audit import digest
from .probe_candidates import read,verify
from .fixed_recovery_dataset import load_fixed_cache
from .structured_features import split_joint,pack_observable
from .robot_kinematics import RobotFK

PAIRED=ROOT/'outputs/effect_learning/paired_fixed_20261003_040014_081930'
CACHE=ROOT/'outputs/feature_caches/fixed_recovery_20261003_031535_304147'
DIAG=ROOT/'outputs/reports/diagnostics/teacher_observability_20261003_044742_113205'
FEATURES=ROOT/'outputs/feature_caches/joint_prefix_20260930_141244_699126'


def main():
    p=read(PAIRED/'protocol.json');verify(p);cp=read(CACHE/'protocol.json')
    load_fixed_cache(CACHE,scope='developer_same_task_scene_holdout')
    if not read(DIAG/'audit.json')['passed']:raise ValueError('S2f audit required')
    expert_rows=[p['rows'][i] for i in p['train_indices'] if p['rows'][i]['kind']=='expert']
    assert len(expert_rows)==572 and len({r['group'] for r in expert_rows})==32
    lookup={(r['group'],r['frame']):i for i,r in enumerate(expert_rows)}
    raw=np.empty((len(expert_rows),28672),np.float32);seen=set();sources=[]
    for record in read(FEATURES/'result.json')['records']:
        source=record['source'];group=source['group']
        if source['kind']!='expert' or group not in {r['group'] for r in expert_rows}:continue
        f=Path(record['file']);assert digest(f)==record['sha256'];sources.append(f)
        with np.load(f) as z:
            for frame,joint in zip(z['frames'],z['joint']):
                key=(group,int(frame))
                if key not in lookup:continue
                i=lookup[key];assert i not in seen;raw[i]=joint;seen.add(i)
    assert seen==set(range(572))
    run=new_run('structured_readout',group=('feature_caches',));print('RUN',run,flush=True)
    files=[PAIRED/'protocol.json',CACHE/'protocol.json',CACHE/'audit.json',CACHE/'features.npz',DIAG/'audit.json',FEATURES/'result.json',
        ROOT/'cx_tta/structured_features.py',Path(__file__),ROOT/'cx_tta/structured_readout_probe.py',
        ROOT/'cx_tta/train_structured_probe.py',ROOT/'cx_tta/audit_structured_probe.py',ROOT/'scripts/train_structured_probe.sh',*sources]
    # Freeze design/input identities before fitting expert-only coordinates.
    write_json(run/'design.json',dict(status='frozen_before_projection',fit_expert_rows=expert_rows,channel_dim=32,
        fit_scope='572 original gate-fit expert rows/32scenes only,14tokens per row; no anchors/recovery/developer/validation',
        method='Shared PCA32 over token channels, randomized solver seed0; scale by expert projected channel std floor.05. '
            'Preserve camera order head/right/left and row-major2x2 positions plus text mean/last.',
        inputs='Current and real32-step past visual/text/proprio, history availability and command-FK poses of both arms. '
            'No task name/scene/branch/native counter/teacher plan/target/GT fed forward.',
        experiment='Three same-task scene omissions x uniform/conditional/visual_disabled x seeds0,1,2. '
            'Same model and actual initialization within each scene+seed, same train rows/loss/600updates/lr.0003. '
            'Diagnostic head only; no new action policy/router/TTA. No candidate/checkpoint selection.',
        source_sha256={str(f):digest(f) for f in files}))
    v,t=split_joint(raw);tokens=np.concatenate([v.reshape(-1,12,2048),t],axis=1).reshape(-1,2048)
    pca=PCA(n_components=32,svd_solver='randomized',random_state=0).fit(tokens)
    projected=pca.transform(tokens).astype(np.float32)
    pre=dict(mean=pca.mean_.astype(np.float32),components=pca.components_.astype(np.float32),scale=np.maximum(projected.std(0),.05))
    np.savez_compressed(run/'projection.npz',**pre)
    with np.load(CACHE/'features.npz') as z:joint=z['joint'].copy();state=z['state'].copy()
    curr=np.array(cp['feature_current_indices']);history=np.array(cp['feature_history_indices']);prev=history[curr];has=prev>=0
    if not has.all():raise ValueError('Fixed recovery must have actual fork history')
    for now,before in zip(curr,prev):
        x,y=cp['feature_rows'][now],cp['feature_rows'][before]
        assert before<now and x['group']==y['group'] and x['scene_group']==y['scene_group'] and x['native_count']-y['native_count']==32
    robots=[RobotFK(read(Path(p['config'])),side) for side in ('left','right')]
    out=pack_observable(joint[curr],state[curr],joint[prev],state[prev],has,pre,robots)
    np.savez_compressed(run/'inputs.npz',**out)
    # Independent shape/order readback, actual previous features, scale and FK.
    visual,text=split_joint(joint);reconstructed=np.concatenate([visual.reshape(len(joint),12,2048),text],1).reshape(len(joint),28672)
    np.testing.assert_array_equal(reconstructed,joint)
    direct=(joint.astype(np.float32).reshape(-1,14,2048)@pre['components'].T-pre['mean'][None,None]@pre['components'].T)/pre['scale']
    np.testing.assert_array_equal(out['visual'],direct[curr,:12].reshape(23,3,2,2,32))
    np.testing.assert_array_equal(out['past_text'],direct[prev,12:])
    np.testing.assert_allclose(out['state'][:,[6,13]],state[curr][:,[6,13]],atol=0)
    for i in range(23):
        for side,start,arm in zip(('left','right'),(0,7),range(2)):
            expected=robots[arm].pose(state[curr[i],start:start+6].astype(float))
            np.testing.assert_allclose(out['ee_pose'][i,arm,:3]*.1,expected[:3,3],atol=1e-7)
    verify(read(run/'design.json'));verify(p)
    protocol=dict(status='complete_interface',rows=cp['rows'],cache=str(CACHE),paired=str(PAIRED),diagnostic=str(DIAG),
        fit_experts=expert_rows,channel_dim=32,explained_variance=float(pca.explained_variance_ratio_.sum()),
        source_sha256={**read(run/'design.json')['source_sha256'],str(run/'design.json'):digest(run/'design.json'),
            str(run/'inputs.npz'):digest(run/'inputs.npz'),str(run/'projection.npz'):digest(run/'projection.npz')},
        forward_keys=list(out),input_rows=23,independent_scenes=4,new_policy_training=False)
    write_json(run/'protocol.json',protocol)
    write_json(run/'interface_audit.json',dict(passed=True,raw_roundtrip_exact=True,projected_order_exact=True,
        causal_history_exact=True,state_pi_gripper_contract=True,command_fk_parity=True,all_input_fields_observable=True,
        projection_fit_expert_rows=572,projection_fit_expert_scenes=32,policy_updates=0,
        source_sha256={str(run/'protocol.json'):digest(run/'protocol.json')}))
    print('INTERFACE_PASSED',run,'variance',protocol['explained_variance'],flush=True)

if __name__=='__main__':
    threadpool_limits(2);main()
