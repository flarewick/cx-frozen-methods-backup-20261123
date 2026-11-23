"""Common pulse-start training interface. Labels and future validity stay offline."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_cartesian_pulses import audit_run
from .object_effect_labels import entity_effects
from .paths import ROOT,new_run,output_path,write_json
from .robot_kinematics import RobotFK,cartesian_residual
from .short_window_dataset import frame_features


def build(runs,split_plan):
    plan=json.loads(split_plan.read_text())
    split={(r['task'],r['seed']):r['split'] for r in plan['scenes']}
    if len(split)!=len(plan['scenes']):raise ValueError('Repeated scene in split plan')
    records=[];labels=[];contexts=[];actions=[];planned=[];sources=[];seen=set()
    for run in runs:
        run=output_path(run);audit_run(run)
        manifest=json.loads((run/'manifest.json').read_text());result=json.loads((run/'result.json').read_text())
        key=(result['task'],result['seed']);start=manifest['pulse_start']
        identity=(*key,result['chunk'],start)
        if key not in split or identity in seen:raise ValueError('Unknown split or repeated decision point')
        seen.add(identity);config=json.loads((Path(manifest['source'])/'resolved_config.json').read_text())
        robots=[RobotFK(config,side) for side in ('left','right')]
        base_branch=result['branches']['base'];baseframes={f['offset']:f for f in base_branch['frames']}
        current=baseframes[start];history=baseframes.get(start-4)
        current_vector=frame_features(run/current['observed_file'])
        history_vector=frame_features(run/history['observed_file']) if history is not None else np.zeros_like(current_vector)
        with np.load(run/current['observed_file']) as z:obs=dict(z)
        with np.load(run/'checkpoint.npz') as z:base=z['base_actions']
        anchor=obs['command_state'].copy();anchor[:6]=obs['arm_qpos'][0];anchor[7:13]=obs['arm_qpos'][1]
        for side,i,ch in [('left',0,6),('right',1,13)]:
            low,high=config[side+'_embodiment_config']['gripper_scale'];anchor[ch]=(obs['gripper_qpos'][i]-low)/(high-low)
        base_motion=cartesian_residual(np.repeat(anchor[None],32,axis=0),base,robots)
        with np.load(run/'transition_base.npz') as z:base_mask=z['executed_mask'];base_dt=z['physics_dt_s']
        sources.append(dict(run=str(run),manifest_sha256=digest(run/'manifest.json'),result_sha256=digest(run/'result.json'),current_observed_file=str(run/current['observed_file']),current_observed_sha256=current['observed_sha256'],history_observed_file=str(run/history['observed_file']) if history else None,history_observed_sha256=history['observed_sha256'] if history else None))
        for name,branch in result['branches'].items():
            frames={f['offset']:f for f in branch['frames']}
            with np.load(run/f'transition_{name}.npz') as z:arr=dict(z)
            if start:
                past=base[start-4:start]
            else:past=arr['history_actions']
            if past.shape!=(4,14):past=np.zeros((4,14),np.float32)
            for stop in range(start+4,33,4):
                residual=arr['planned_cartesian_residual'].copy();residual[:start]=0;residual[stop:]=0
                base_plan=base_motion.copy();base_plan[:start]=0;base_plan[stop:]=0
                valid=bool(arr['executed_mask'][start:stop].all() and base_mask[start:stop].all() and arr['physics_dt_s'][start:stop].sum()>0 and base_dt[start:stop].sum()>0)
                end,reference=frames[stop],baseframes[stop]
                meta=dict(task=key[0],seed=key[1],group=f'{key[0]}:{key[1]}',chunk=result['chunk'],branch=name,start=start,stop=stop,horizon=stop-start,split=split[key],source_index=len(sources)-1)
                label=dict(counterfactual=entity_effects(reference['physical_offline_only']['offline_only'],end['physical_offline_only']['offline_only'],branch['articulation_limits_offline_only']),visibility_before=current['visibility_offline_only'],visibility_after=end['visibility_offline_only'],base_visibility_after=reference['visibility_offline_only'],full_horizon_valid_offline_only=valid,duration_s_offline_only=float(arr['physics_dt_s'][start:stop].sum()),base_duration_s_offline_only=float(base_dt[start:stop].sum()),contact_before_offline_only=current['contacts_offline_only'],contact_after_offline_only=end['contacts_offline_only'])
                context=np.r_[current_vector,history_vector,float(history is not None),(past/np.array([np.pi]*6+[1.]+[np.pi]*6+[1.])).ravel(),(stop-start)/32]
                contexts.append(context);actions.append(residual.ravel());planned.append(base_plan.ravel());records.append(meta);labels.append(label)
    if not records:raise ValueError('Empty dataset')
    run=new_run('pulse_dataset',group=('effect_learning',))
    np.savez_compressed(run/'inputs_observable.npz',context=np.asarray(contexts,np.float32),planned_base_fk=np.asarray(planned,np.float32),planned_residual_fk=np.asarray(actions,np.float32))
    write_json(run/'labels_offline.json',labels)
    write_json(run/'index.json',dict(status='complete',records=records,sources=sources,inputs_sha256=digest(run/'inputs_observable.npz'),labels_sha256=digest(run/'labels_offline.json'),split_plan=str(split_plan),split_plan_sha256=digest(split_plan),independent_scenes=len({r['group'] for r in records}),scope='CPU learning-interface only: current/history coarse RGB and proprioception; known original-query plan and residual in robot FK coordinates. No object GT, task ID or future realized duration as numeric inputs. Entity associations in labels privileged; not deployable M1 or trained controller. All horizons retained; scene split from explicit frozen plan.'))
    return run


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--split-plan',type=Path,default=ROOT/'configs/intervention_batch1_20260927.json');args=p.parse_args()
    run=build(args.runs,args.split_plan);print(run,flush=True)

if __name__=='__main__':main()
