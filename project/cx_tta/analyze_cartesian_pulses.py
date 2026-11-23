"""Offline mechanism diagnostics; physical labels never become policy inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .audit_cartesian_pulses import audit_run
from .paths import output_path,write_json,new_run
from .robot_kinematics import RobotFK


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4,5,6,3]])


def analyze(run):
    run=output_path(run);audit=audit_run(run)
    result=json.loads((run/'result.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    config=json.loads((Path(manifest['source'])/'resolved_config.json').read_text())
    robots=[RobotFK(config,side) for side in ('left','right')]
    base={f['offset']:f for f in result['branches']['base']['frames']}
    rows=[]
    for name,branch in result['branches'].items():
        if name=='base':continue
        side,axis,sign=name.split('_');i=0 if side=='left' else 1
        direction=np.eye(3)['xyz'.index(axis)]*(1 if sign=='plus' else -1)
        with np.load(run/f'transition_{name}.npz') as z:planned=z['planned_cartesian_residual']
        peak_index=manifest['pulse_start']+3
        planned_delta=planned[peak_index,i*7:i*7+3]*.01
        for frame in branch['frames']:
            offset=frame['offset'];ref=base.get(offset)
            if ref is None:continue
            current=frame['physical_offline_only'];reference=ref['physical_offline_only']
            pose=robots[i].pose(current['arm_qpos'][i]);bpose=robots[i].pose(reference['arm_qpos'][i])
            delta=pose[:3,3]-bpose[:3,3]
            entities={}
            for entity,p in current['offline_only']['object_poses'].items():
                b=reference['offline_only']['object_poses'][entity]
                translation=np.asarray(p[:3])-np.asarray(b[:3])
                entities[entity]=dict(translation_world_m=translation.tolist(),translation_norm_m=float(np.linalg.norm(translation)),rotation_world_rad=(rotation(p)*rotation(b).inv()).as_rotvec().tolist(),rotation_norm_rad=float(np.linalg.norm((rotation(p)*rotation(b).inv()).as_rotvec())))
            joints={key:(np.asarray(q)-np.asarray(reference['offline_only']['articulation_qpos'][key])).tolist() for key,q in current['offline_only']['articulation_qpos'].items()}
            contacts=frame['contacts_offline_only']['pairs']
            rows.append(dict(branch=name,offset=offset,common_prefix=offset<=manifest['pulse_start'],realized_link_delta_world_m=delta.tolist(),realized_along_requested_axis_m=float(delta@direction),realized_direction_cosine=None if np.linalg.norm(delta)<1e-8 else float(delta@direction/np.linalg.norm(delta)),planned_peak_delta_world_m=planned_delta.tolist(),entities_offline_only=entities,articulation_delta_offline_only=joints,contact_body_names_at_snapshot=[pair['entity_names'] for pair in contacts]))
    pulse_peak=manifest['pulse_start']+4
    summary={}
    for offset in (pulse_peak,manifest['pulse_start']+8,32):
        selected=[r for r in rows if r['offset']==offset]
        summary[str(offset)]=dict(realized_link_norm_m={r['branch']:float(np.linalg.norm(r['realized_link_delta_world_m'])) for r in selected},max_entity_translation_m={entity:max(r['entities_offline_only'][entity]['translation_norm_m'] for r in selected) for entity in selected[0]['entities_offline_only']})
    return dict(status='complete',audit_status=audit['status'],task=result['task'],seed=result['seed'],chunk=result['chunk'],pulse_start=manifest['pulse_start'],run=str(run),rows=rows,summary=summary,scope='One-scene offline branch-minus-base effects and realized link pose deviation, not success rate or transfer evidence. Planned peak indexed at pulse_start+3; snapshot after pulse_start+4 actions. Contact lists are instantaneous and include table/object contacts.')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.run)
    target=args.run/'mechanism_analysis.json'
    if target.exists():target=new_run('pulse_analysis',group=('reports','analyses'))/'result.json'
    write_json(target,result)
    print('analysis_file='+str(target),flush=True)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
