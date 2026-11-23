"""Oracle object inputs and shared seven-component effect labels; no task IDs as features."""
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .cache_siglip import setup_imports
from .effect_learning_probe import weights,shuffle_within_checkpoint
from .object_alignment import pool_object
from .paths import ROOT,new_run,write_json
from .robot_kinematics import RobotFK,cartesian_residual
from .semantic_effect_probe import squared_distances,rbf,fit_predict
from .short_window_probe import targets,ALPHA
from .shared_effect_model import project_relation


def main():
    run=new_run('shared_dataset',group=('effect_learning',));print(run,flush=True)
    fk_audit=json.loads((ROOT/'outputs/reports/audits/fk_camera_check_20260928.json').read_text())
    if fk_audit['status']!='passed':raise ValueError('FK failed camera verification')
    setup_imports(run)
    from openpi.shared.image_tools import resize_with_pad
    cache=ROOT/'outputs/feature_caches/short_siglip_train_v1_gpu'
    cached=json.loads((cache/'result.json').read_text());lookup={(r['job'],r['offset']):r for r in cached['records']}
    horizons=json.loads((ROOT/'outputs/reports/audits/cumulative_probes_20260928.json').read_text())['records']
    first=json.loads((Path(horizons[0]['dataset'])/'index.json').read_text())
    global_features={};object_features={};cartesian={};provenance=[];relations={};base_motion={};proprio={}
    for source in first['sources']:
        path=Path(source['run']);result=json.loads((path/'result.json').read_text());manifest=json.loads((path/'manifest.json').read_text())
        if digest(path/'result.json')!=source['result_sha256']:raise ValueError('Source changed')
        config_path=Path(manifest['source'])/'resolved_config.json';config=json.loads(config_path.read_text());robots=[RobotFK(config,side) for side in ('left','right')]
        provenance.append(dict(job=source['id'],config_sha256=digest(config_path),urdf_sha256=digest(robots[0].urdf)))
        frames={f['offset']:f for f in result['branches']['base']['frames']};rel={};states=[];mapping=result['branches']['base']['entity_id_map_offline'];vf=[];of={name:[] for name in mapping}
        for offset in (0,-8):
            frame=frames[offset];row=lookup[source['id'],offset]
            with np.load(path/frame['observed_file']) as obs:
                obs=dict(obs)
            poses=[robots[i].pose(obs['arm_qpos'][i]) for i in range(2)]
            states.append(np.r_[obs['arm_qpos'].ravel()/np.pi,obs['gripper_qpos']])
            if offset==0:
                anchor=obs['command_state'].copy()
                anchor[:6]=obs['arm_qpos'][0];anchor[7:13]=obs['arm_qpos'][1]
                for side,i,channel in [('left',0,6),('right',1,13)]:
                    low,high=config[side+'_embodiment_config']['gripper_scale'];anchor[channel]=(obs['gripper_qpos'][i]-low)/(high-low)

            if digest(Path(row['file']))!=row['sha256'] or digest(path/frame['offline_file'])!=frame['offline_sha256']:raise ValueError('Feature/mask changed')
            with np.load(row['file']) as z:pooled=z['pooled']
            v=pooled.reshape(3,-1).astype(float);vf.append((v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)).ravel())
            with np.load(path/frame['offline_file']) as z:maps=z['actor_ids']
            for name,ids in mapping.items():
                masks=[]
                for raw in maps:
                    mask=np.repeat((np.isin(raw,ids).astype(np.uint8)*255)[...,None],3,axis=-1)
                    masks.append(np.asarray(resize_with_pad(mask,224,224))[...,0].astype(float)/255)
                vector,geometry=pool_object(pooled,np.stack(masks));of[name].append((vector,geometry))
                relation=[]
                for camera in range(3):
                    for pose in poses:
                        relation.extend(project_relation(pose,obs['intrinsic_cv'][camera],obs['extrinsic_cv'][camera],maps[camera].shape,geometry[camera*4:(camera+1)*4]))
                rel.setdefault(name,[]).append(np.r_[geometry,relation])
        global_features[source['id']]=np.concatenate(vf)
        for name,pairs in of.items():
            object_features[source['id'],name]=(np.concatenate([p[0] for p in pairs]),np.concatenate([p[1] for p in pairs]))
            relations[source['id'],name]=np.concatenate(rel[name])
        proprio[source['id']]=np.concatenate(states)
        with np.load(path/'checkpoint.npz') as z:base=z['base_actions']
        base_motion[source['id']]=cartesian_residual(np.repeat(anchor[None],32,axis=0),base,robots)
        for name in result['branches']:
            with np.load(path/f'transition_{name}.npz') as z:proposal=z['proposed_actions']
            cartesian[source['id'],name]=cartesian_residual(base,proposal,robots)
        print('PREPARED '+source['id'],flush=True)
    records=[];visual=[];geometry=[];action=[];targets_out=[];masks=[]
    scale=np.array([.01]*3+[.1]*3+[.1])
    for h in horizons:
        dataset=Path(h['dataset']);index=json.loads((dataset/'index.json').read_text());labels=json.loads((dataset/'labels_offline.json').read_text())
        if digest(dataset/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Labels changed')
        for row,label in zip(index['records'],labels):
            if row['branch']=='base':continue
            job=row['job'];branch=row['branch'];horizon=h['horizon']
            for entity,rigid in label['counterfactual']['rigid'].items():
                y=np.zeros(7);mask=np.zeros(7,bool);y[:3]=rigid['translation_world_m'];y[3:6]=rigid['rotation_world_rad'];mask[:6]=True
                if entity in label['counterfactual']['articulated']:
                    joint=label['counterfactual']['articulated'][entity]
                    if len(joint['delta_range_fraction'])!=1:raise ValueError('Diagnostic supports one articulation DOF only')
                    y[6]=joint['delta_range_fraction'][0];mask[6]=joint['range_valid_mask'][0]
                visible=all(label[k][entity]['visible_in_any_camera'] for k in ('visibility_before','visibility_after','base_visibility_after'))
                residual=cartesian[job,branch].copy();residual[horizon:]=0
                planned=base_motion[job].copy();planned[horizon:]=0
                visual.append(object_features[job,entity][0])
                geometry.append(np.r_[relations[job,entity],proprio[job],planned.reshape(4,8,14).mean(axis=1).ravel(),horizon/32])
                action.append(residual.ravel());targets_out.append(y/scale);masks.append(mask)
                records.append(dict(row,entity=entity,horizon=horizon,visible_supervision_only=visible))
    np.savez_compressed(run/'inputs_oracle.npz',vision=np.array(visual,np.float32),geometry=np.array(geometry,np.float32),action=np.array(action,np.float32))
    np.savez_compressed(run/'labels_offline.npz',target=np.array(targets_out,np.float32),mask=np.array(masks),scale=scale)
    write_json(run/'index.json',dict(status='complete',records=records,provenance=provenance,inputs_sha256=digest(run/'inputs_oracle.npz'),labels_sha256=digest(run/'labels_offline.npz'),scope='PRIVILEGED initial/history object masks and association; no task/entity ID or future truth as numeric features. Shared rigid 6D plus single articulated DOF supervision. All scenes from original train only.'))
    print(f'COMPLETE rows={len(records)} output={run}',flush=True)

if __name__=='__main__':main()
