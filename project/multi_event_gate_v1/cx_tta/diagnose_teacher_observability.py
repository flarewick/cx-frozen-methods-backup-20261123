"""S2f: immutable teacher consistency, phase errors, and scene-held-out linear probes.

All privileged values are diagnostic targets only. No adapter/policy updates.
"""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from .paths import ROOT,new_run,write_json
from .audit import digest
from .probe_candidates import read,verify
from .paired_fixed_data import load as load_paired
from .robot_kinematics import RobotFK
from .audit_pose_capacity import numpy_fk,angles
from .recovery_fit_metrics import endpoints

PAIRED=ROOT/'outputs/effect_learning/paired_fixed_20261003_040014_081930'
CACHE=ROOT/'outputs/feature_caches/fixed_recovery_20261003_031535_304147'
STAGES={'turn_switch':('closure','approach','contact','final_closure'),
        'place_shoe':('closure','lift','preplace','place','release')}


def pose7(x):
    t=np.eye(4);t[:3,3]=x[:3];t[:3,:3]=Rotation.from_quat(np.array(x[3:])[[1,2,3,0]]).as_matrix();return t


def balanced(rows,ids):
    scenes={rows[i]['scene_group'] for i in ids};out=[]
    for i in ids:
        scene=rows[i]['scene_group'];branch=rows[i]['group']
        branches={rows[j]['group'] for j in ids if rows[j]['scene_group']==scene}
        count=sum(rows[j]['group']==branch for j in ids)
        out.append(1/(len(scenes)*len(branches)*count))
    return np.array(out)


def transform_blocks(blocks,train):
    """Training-only scalar RMS per block; no held moments, target info or PCA refit."""
    output=[]
    for b in blocks:
        b=np.asarray(b,np.float64);mean=b[train].mean(0)
        rms=max(float(np.sqrt(np.mean((b[train]-mean)**2))),1e-3)
        output.append((b-mean)/(rms*np.sqrt(b.shape[1])))
    return np.concatenate(output,1)/np.sqrt(len(output))


def ridge(x,y,train,weights):
    """Fixed alpha=1, weighted centered dual ridge; y is indexed at train only."""
    xt=x[train];yt=y[train];mx=weights@xt;my=weights@yt
    w=np.sqrt(weights);a=(xt-mx)*w[:,None];b=(yt-my)*w[:,None]
    coef=np.linalg.solve(a@a.T+np.eye(len(train)),b)
    return ((x-mx)@a.T)@coef+my


def inspect_teachers(inventory,robots):
    branches=[];traces={};files=[]
    for attempt in inventory['attempts']:
        if attempt['kind'] not in ('switch','shoe_teacher'):continue
        run=Path(attempt['run']);result=read(run/'result.json');fork=read(run/'fork.json')
        trace=[__import__('json').loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
        traces[str(run)]=trace;files += [run/'result.json',run/'fork.json',run/'trace.jsonl',run/'fork.npz']
        task=attempt['task'];stages=STAGES[task];plans=[]
        for f in sorted(run.glob('plan_*.npz')):
            plan=int(f.stem.split('_')[-1]);assert plan<len(stages)
            with np.load(f) as z:d=z['dense_commands'].copy();indices=z['indices'].copy()
            active=[side for side,start in zip(('left','right'),(0,7)) if np.max(np.ptp(d[:,start:start+6],axis=0))>1e-5]
            expected=[] if stages[plan] in ('closure','release','final_closure') else ['right' if task=='turn_switch' else 'left']
            assert active==expected,(run,plan,active,expected)
            executed=[t for t in trace if t['plan']==plan]
            for t in executed:np.testing.assert_array_equal(np.asarray(t['action'],np.float32),d[t['dense_index']])
            item=dict(plan=plan,stage=stages[plan],dense_count=len(d),sampled_count=len(indices),executed_count=len(executed),
                planned_endpoint_executed=bool(executed and executed[-1]['dense_index']==indices[-1]),
                moving_arms=active,gripper_start=d[0,[6,13]].tolist(),gripper_end=d[-1,[6,13]].tolist(),
                endpoint_poses=[robot.pose(d[-1,start:start+6]).tolist() for robot,start in zip(robots,(0,7))])
            if plan==0:
                item['max_arm_motion']=float(np.ptp(d[:,[0,1,2,3,4,5,7,8,9,10,11,12]],axis=0).max())
                item['near_closed_right_fraction']=float(np.mean(d[indices,13]<=.01))
            plans.append(item);files.append(f)
        with np.load(run/'fork.npz') as z:state=z['command_state'].copy()
        closure_obs=run/'observation_0020.npz';closure_change=None
        if closure_obs.exists():
            with np.load(run/'fork.npz') as a,np.load(closure_obs) as b:
                closure_change=dict(arm_state_max_abs=float(np.max(np.abs(a['command_state'][[0,1,2,3,4,5,7,8,9,10,11,12]]-b['command_state'][[0,1,2,3,4,5,7,8,9,10,11,12]]))),
                    right_gripper_before=float(a['command_state'][13]),right_gripper_after=float(b['command_state'][13]),
                    rgb_mae_255={k:float(np.mean(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in ('head_rgb','right_rgb','left_rgb')})
            files.append(closure_obs)
        branch=dict(**attempt,plans=plans,closure_observation_change=closure_change,fork_command_state=state.tolist(),
            instruction=read(run/'manifest.json')['instruction'])
        if task=='turn_switch':
            switch=fork['diagnostics']['physical']['offline_only']['object_poses']['switch'];branch['switch_root_pose']=switch
            branch['official_owner']='right' if -pose7(switch)[0,0]>0 else 'left';assert branch['official_owner']=='right'
        branches.append(branch)
    pairs=[]
    for a,b in combinations(branches,2):
        if a['task']!='turn_switch' or b['task']!='turn_switch' or a['seed']!=b['seed']:continue
        for plan in (1,2):
            pa=next((p for p in a['plans'] if p['plan']==plan),None);pb=next((p for p in b['plans'] if p['plan']==plan),None)
            if pa is None or pb is None:continue
            x,y=[np.array(v['endpoint_poses'][1]) for v in (pa,pb)]
            pairs.append(dict(scene=a['seed'],points=[a['point'],b['point']],stage=pa['stage'],
                both_qualified=a['qualified'] and b['qualified'],position_m=float(np.linalg.norm(x[:3,3]-y[:3,3])),angle_deg=float(angles(x,y)),
                both_planned_endpoints_executed=pa['planned_endpoint_executed'] and pb['planned_endpoint_executed']))
    return branches,traces,pairs,files


def phase_errors(p,a,result,cp,traces,robots):
    ids=np.array(p['new_indices']);n=len(ids);rows=cp['rows'];phase=np.full((n,32),-1,int)
    for i,row in enumerate(rows):
        trace=traces[row['source']];r=read(Path(row['source'])/'result.json');offset=row['native_count']-r['prefix_actions']
        count=int(a['valid'][ids[i]].sum());actual=np.array([t['action'] for t in trace[offset:offset+count]],np.float32)
        np.testing.assert_array_equal(actual,a['target'][ids[i],:count]);phase[i,:count]=[t['plan'] for t in trace[offset:offset+count]]
    target=a['target'][ids];valid=a['valid'][ids];tp=endpoints(target,robots);reports=[]
    variants=[('base',None,None,None,a['base'][ids])]
    for r in result['reports']:
        with np.load(r['artifact']) as z:pred=z['predicted'][ids]
        variants.append((r['condition'],r['encoding'],r['seed'],'raw',pred))
    for condition,encoding,seed,variant,pred in variants:
        pp=endpoints(pred,robots);angle=angles(pp,tp);position=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1)
        for scene in sorted({r['scene_group'] for r in rows}):
            wanted=np.array([r['scene_group']==scene for r in rows]);task=next(r['task'] for r in rows if r['scene_group']==scene)
            if condition not in ('base','include_all','omit_switch_'+scene.split(':')[-1]):continue
            for stage_id,stage in enumerate(STAGES[task]):
                mask=(phase==stage_id)&wanted[:,None]&valid
                if not mask.any():continue
                arms={}
                for arm,start in enumerate((0,7)):
                    arms[('left','right')[arm]]=dict(angle_deg=float(angle[...,arm][mask].mean()),
                        position_m=float(position[...,arm][mask].mean()),
                        pose_fraction=float(((angle[...,arm]<=3)&(position[...,arm]<=.01))[mask].mean()),
                        arm_rmse_rad=float(np.sqrt(np.mean((pred[...,start:start+6][mask]-target[...,start:start+6][mask])**2))),
                        gripper_mae=float(np.abs(pred[...,start+6][mask]-target[...,start+6][mask]).mean()))
                reports.append(dict(condition=condition,encoding=encoding,seed=seed,scene=scene,stage=stage,actions=int(mask.sum()),arms=arms))
    return phase,reports


def probes(p,a,context,cp,branches,phase):
    rows=cp['rows'];which=np.array([i for i,r in enumerate(rows) if r['task']=='turn_switch']);cache_rows=[rows[i] for i in which]
    global_ids=np.array(p['new_indices'])[which];current_ids=np.array(cp['feature_current_indices'])[which]
    history=np.array(cp['feature_history_indices']);previous_ids=history[current_ids];assert np.all(previous_ids>=0)
    with np.load(CACHE/'features.npz') as z:joint=z['joint'].astype(np.float64);state=z['state'].astype(np.float64)
    state[:,[0,1,2,3,4,5,7,8,9,10,11,12]]/=np.pi
    with np.load(CACHE/'data.npz') as z:coarse=z['coarse'][which].copy()
    semantic=context[global_ids]
    representations={
        'proprio_history':[state[current_ids],state[previous_ids]],
        'semantic_pca_history':[semantic[:,:64],semantic[:,64:78],semantic[:,78:142],semantic[:,142:156]],
        'raw_joint_history':[joint[current_ids],state[current_ids],joint[previous_ids],state[previous_ids]],
        'coarse_history':[coarse[:,:64],coarse[:,64:78],coarse[:,78:142],coarse[:,142:156]]}
    root=[];pregrasp=[];grip=[];state_open=[]
    for i,row in zip(which,cache_rows):
        branch=next(b for b in branches if b['run']==row['source']);root.append(branch['switch_root_pose'][:3])
        pregrasp.append(np.array(next(p for p in branch['plans'] if p['plan']==1)['endpoint_poses'][1])[:3,3])
        valid=a['valid'][p['new_indices'][i]];grip.append([float(a['target'][p['new_indices'][i],:,13][valid].mean())])
        with np.load(row['observation']) as z:state_open.append(float(z['command_state'][13]>.5))
    targets={'root_world_xyz':np.array(root),'pregrasp_world_xyz':np.array(pregrasp),'right_grip_chunk_mean':np.array(grip)}
    stage=phase[which,0];results=[];isolation=[]
    for scene in sorted({r['scene_group'] for r in cache_rows}):
        train=np.array([i for i,r in enumerate(cache_rows) if r['scene_group']!=scene]);held=np.array([i for i,r in enumerate(cache_rows) if r['scene_group']==scene])
        assert not ({cache_rows[i]['scene_group'] for i in train}&{cache_rows[i]['scene_group'] for i in held})
        weights=balanced(cache_rows,train);hw=balanced(cache_rows,held)
        classes=sorted(set(stage[train].tolist()));onehot=np.eye(4)[stage]
        majority=max(classes,key=lambda c:float(weights[stage[train]==c].sum()))
        for name,blocks in representations.items():
            x=transform_blocks(blocks,train);entry=dict(held_scene=scene,representation=name,train_rows=len(train),held_rows=len(held),regression={})
            for label,y in targets.items():
                pred=ridge(x,y,train,weights);prior=weights@y[train]
                error=np.linalg.norm(pred[held]-y[held],axis=1);base=np.linalg.norm(prior-y[held],axis=1)
                entry['regression'][label]=dict(held_error=float(hw@error),prior_error=float(hw@base),
                    train_error=float(weights@np.linalg.norm(pred[train]-y[train],axis=1)),predicted=pred[held].tolist(),target=y[held].tolist())
                altered=y.copy();altered[held]=123.0
                assert np.array_equal(pred,ridge(x,altered,train,weights))
            scores=ridge(x,onehot,train,weights);guess=np.argmax(scores,axis=1)
            dist=((x[held,None]-x[train][None])**2).sum(-1);nearest=train[np.argmin(dist,axis=1)]
            entry['phase']=dict(target=stage[held].tolist(),ridge=guess[held].tolist(),nearest=stage[nearest].tolist(),
                accuracy=float(hw@(guess[held]==stage[held])),nearest_accuracy=float(hw@(stage[nearest]==stage[held])),
                majority_accuracy=float(hw@(majority==stage[held])),held_classes=sorted(set(stage[held].tolist())),train_classes=classes)
            entry['open_grip_coverage']=dict(train_open_rows=int(np.sum(np.array(state_open)[train])),held_open_rows=int(np.sum(np.array(state_open)[held])))
            results.append(entry)
        isolation.append(dict(scene=scene,whole_scene_disjoint=True,poisoned_held_targets_no_effect=True))
    return results,isolation,targets,which


def main():
    p,a,ctx,s=load_paired(PAIRED);cp=read(CACHE/'protocol.json');inventory=read(CACHE/'inventory.json')
    assert read(PAIRED/'candidate_audit.json')['passed'] and read(CACHE/'audit.json')['passed']
    robots=[RobotFK(read(Path(p['config'])),side) for side in ('left','right')]
    run=new_run('teacher_observability',group=('reports','diagnostics'));print('RUN',run,flush=True)
    # Freeze fixed design before inspecting teacher/probe outcomes.
    write_json(run/'protocol.json',dict(status='frozen_before_analysis',paired=str(PAIRED),cache=str(CACHE),
        method='Read all14 teacher attempts, retain negative/censored cases. Compare same-scene planned endpoints, not success-censored terminal points. '
            'Per-native-action stage errors on frozen24models. Whole-scene diagnostic probes only on20cached switch rows/3scenes. '
            'Four fixed feature families, train-only block RMS, weighted dual ridge alpha1 and1NN phase check; no hyperparameter selection. '
            'Targets: privileged root/pregrasp world xyz, mean right grip, current teacher plan. No target/phase/seed as input. '
            'Equal scene/branch/row weighting; compare train-only mean/majority baselines. '
            'World-position error equals relative-position error after expressing both predicted and true point in same legal command-FK frame.',
        limits='Three scene/appearance/instruction bundles, no independent repeats pernewfork; correlations not causal identifiability. '
            'Shoe only one qualified scene: descriptive stages only. Failed teachers never become positive fit labels. '
            'No policy training, no new observations, no scene replay, no router/TTA.',
        source_sha256={str(f):digest(f) for f in (Path(__file__),PAIRED/'protocol.json',PAIRED/'result.json',PAIRED/'data.npz',
            PAIRED/'candidate_audit.json',CACHE/'protocol.json',CACHE/'audit.json',CACHE/'inventory.json',CACHE/'features.npz',CACHE/'data.npz')}
    ))
    branches,traces,pairs,files=inspect_teachers(inventory,robots)
    phase,errors=phase_errors(p,a,read(PAIRED/'result.json'),cp,traces,robots)
    probe,isolation,targets,which=probes(p,a,ctx,cp,branches,phase)
    # Raw observations remain immutable, no RGB or backbone copies.
    verify(read(run/'protocol.json'));verify(p)
    write_json(run/'result.json',dict(status='complete',branches=branches,same_scene_endpoint_pairs=pairs,phase_errors=errors,
        probes=probe,probe_isolation=isolation,source_sha256={str(f):digest(f) for f in sorted(set(files))},
        privileged_inputs_to_policy=False,optimizer_steps=0,new_observations=0))
    lines=['# S2f teacher consistency and legal-observation probes','',
        'CPU-only diagnostic. Frozen adapter predictions, no policy updates; 3 switch scenes only.', '',
        '|Held scene|Features|Root error / prior (cm)|Pregrasp error / prior (cm)|Phase accuracy / majority|',
        '|---|---|---|---|---|']
    for r in probe:
        v=r['regression'];root=v['root_world_xyz'];pre=v['pregrasp_world_xyz'];ph=r['phase']
        lines.append(f"|{r['held_scene']}|{r['representation']}|{100*root['held_error']:.2f} / {100*root['prior_error']:.2f}|{100*pre['held_error']:.2f} / {100*pre['prior_error']:.2f}|{ph['accuracy']:.3f} / {ph['majority_accuracy']:.3f}|")
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',run,flush=True)

if __name__=='__main__':
    threadpool_limits(2);main()
