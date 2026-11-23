"""Bridge pulse data to the same shared oracle-object diagnostic network."""
import argparse,json
from pathlib import Path
import numpy as np
from .audit import digest
from .cache_siglip import setup_imports
from .object_alignment import pool_object
from .paths import ROOT,new_run,write_json
from .robot_kinematics import RobotFK
from .shared_effect_model import project_relation


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    index=json.loads((a.dataset/'index.json').read_text());labels=json.loads((a.dataset/'labels_offline.json').read_text())
    if digest(a.dataset/'inputs_observable.npz')!=index['inputs_sha256'] or digest(a.dataset/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Dataset changed')
    cm=json.loads((a.cache/'manifest.json').read_text());cached=json.loads((a.cache/'result.json').read_text())
    if cm['dataset_index_sha256']!=digest(a.dataset/'index.json') or cached['status']!='complete' or cached['limited']:raise ValueError('Wrong or incomplete semantic cache')
    lookup={(r['source_index'],r['role']):r for r in cached['records']}
    if len(lookup)!=2*len(index['sources']) or len(lookup)!=len(cached['records']):raise ValueError('Missing or repeated cache frames')
    run=new_run('shared_pulse_dataset',group=('effect_learning',));print(run,flush=True);setup_imports(run)
    from openpi.shared.image_tools import resize_with_pad
    prepared={};provenance=[]
    for source_index,source in enumerate(index['sources']):
        path=Path(source['run']);result=json.loads((path/'result.json').read_text());manifest=json.loads((path/'manifest.json').read_text())
        if digest(path/'result.json')!=source['result_sha256'] or digest(path/'manifest.json')!=source['manifest_sha256']:raise ValueError('Run changed')
        config_path=Path(manifest['source'])/'resolved_config.json';config=json.loads(config_path.read_text());robots=[RobotFK(config,s) for s in ('left','right')]
        mapping=result['branches']['base']['entity_id_map_offline'];frames={f['offset']:f for f in result['branches']['base']['frames']}
        features={e:[] for e in mapping};relations={e:[] for e in mapping};states=[]
        for role,offset in [('current',manifest['pulse_start']),('history',manifest['pulse_start']-4)]:
            frame=frames[offset];row=lookup[source_index,role]
            if row['source']!=str(path/frame['observed_file']) or row['source_sha256']!=frame['observed_sha256'] or digest(path/frame['observed_file'])!=frame['observed_sha256'] or digest(Path(row['file']))!=row['sha256'] or digest(path/frame['offline_file'])!=frame['offline_sha256']:raise ValueError('Feature/frame/mask mismatch')
            with np.load(path/frame['observed_file']) as z:obs=dict(z)
            with np.load(row['file']) as z:
                pooled=z['pooled']
                if str(z['source_sha256'])!=row['source_sha256'] or pooled.shape!=(3,4,4,1152) or not np.isfinite(pooled).all():raise ValueError('Invalid feature values')
            with np.load(path/frame['offline_file']) as z:maps=z['actor_ids']
            poses=[robots[i].pose(obs['arm_qpos'][i]) for i in range(2)];states.append(np.r_[obs['arm_qpos'].ravel()/np.pi,obs['gripper_qpos']])
            for entity,ids in mapping.items():
                masks=[]
                for raw in maps:
                    mask=np.repeat((np.isin(raw,ids).astype(np.uint8)*255)[...,None],3,axis=-1)
                    masks.append(np.asarray(resize_with_pad(mask,224,224))[...,0].astype(float)/255)
                vector,geometry=pool_object(pooled,np.stack(masks));features[entity].append(vector)
                relation=[]
                for camera in range(3):
                    for pose in poses:relation.extend(project_relation(pose,obs['intrinsic_cv'][camera],obs['extrinsic_cv'][camera],maps[camera].shape,geometry[camera*4:(camera+1)*4]))
                relations[entity].append(np.r_[geometry,relation])
        for entity in mapping:prepared[source_index,entity]=(np.concatenate(features[entity]),np.r_[np.concatenate(relations[entity]),np.concatenate(states)])
        provenance.append(dict(source_index=source_index,config_sha256=digest(config_path),urdf_sha256=digest(robots[0].urdf)))
        print('PREPARED '+str(source_index),flush=True)
    with np.load(a.dataset/'inputs_observable.npz') as z:base=z['planned_base_fk'];residual=z['planned_residual_fk']
    records=[];vision=[];geometry=[];actions=[];targets=[];masks=[];scale=np.array([.01]*3+[.1]*3+[.1])
    for i,(row,label) in enumerate(zip(index['records'],labels)):
        if row['branch']=='base':continue
        # Align the known action plan to the actual common pulse origin; zero-pad to 32.
        start=row['start'];b=np.zeros((32,14));r=np.zeros((32,14));b[:32-start]=base[i].reshape(32,14)[start:];r[:32-start]=residual[i].reshape(32,14)[start:]
        valid=label['full_horizon_valid_offline_only']
        for entity,rigid in label['counterfactual']['rigid'].items():
            y=np.zeros(7);mask=np.zeros(7,bool);y[:3]=rigid['translation_world_m'];y[3:6]=rigid['rotation_world_rad'];mask[:6]=valid
            if entity in label['counterfactual']['articulated']:
                joint=label['counterfactual']['articulated'][entity]
                if len(joint['delta_range_fraction'])!=1:raise ValueError('Only single-DOF articulation currently supported')
                y[6]=joint['delta_range_fraction'][0];mask[6]=valid and joint['range_valid_mask'][0]
            visible=valid and all(label[k][entity]['visible_in_any_camera'] for k in ('visibility_before','visibility_after','base_visibility_after'))
            v,g=prepared[row['source_index'],entity]
            vision.append(v);geometry.append(np.r_[g,b.reshape(4,8,14).mean(axis=1).ravel(),row['horizon']/32]);actions.append(r.ravel());targets.append(y/scale);masks.append(mask)
            records.append(dict(row,job=f"pulse_source_{row['source_index']:04d}",entity=entity,visible_supervision_only=visible,full_horizon_valid_offline_only=valid))
    np.savez_compressed(run/'inputs_oracle.npz',vision=np.asarray(vision,np.float32),geometry=np.asarray(geometry,np.float32),action=np.asarray(actions,np.float32))
    np.savez_compressed(run/'labels_offline.npz',target=np.asarray(targets,np.float32),mask=np.asarray(masks),scale=scale)
    write_json(run/'index.json',dict(status='complete',records=records,provenance=provenance,inputs_sha256=digest(run/'inputs_oracle.npz'),labels_sha256=digest(run/'labels_offline.npz'),dataset=str(a.dataset),dataset_index_sha256=digest(a.dataset/'index.json'),cache=str(a.cache),cache_manifest_sha256=digest(a.cache/'manifest.json'),code_sha256=digest(Path(__file__)),scope='Privileged current/history object masks and association; same shared 7-output model interface, no task/entity IDs or future realized state as numeric inputs. Incomplete horizons retained with false target masks. All 4/8/12/16/20/24 action horizons retained. Current/history separated by 4 actions, pulse-relative action plan zero-padded to 32. No instruction embedding or deployable controller.'))
    print(f'COMPLETE rows={len(records)} output={run}',flush=True)

if __name__=='__main__':main()
