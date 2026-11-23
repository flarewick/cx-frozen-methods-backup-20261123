"""Planned FK direction metrics on held demonstration commands, not object effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .robot_kinematics import RobotFK, cartesian_residual
from .paths import write_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());rows=protocol['rows'];cache=Path(protocol['cache']);features=json.loads((cache/'result.json').read_text())
    sources={r['source']['group']:r['source'] for r in features['records'] if r['source']['kind']=='expert'}
    configs={};labels={}
    for group,source in sources.items():
        if source['pilot_split']!='inner_held':continue
        cpath=Path(source['collection'])/'resolved_config.json'
        if digest(cpath)!=source['config_sha256'] or digest(Path(source['label']))!=source['label_sha256']:raise ValueError('Changed geometry source')
        configs[group]=[RobotFK(json.loads(cpath.read_text()),s) for s in ('left','right')]
        with np.load(source['label']) as z:
            for frame,y,m in zip(z['source_frame_index'],z['expert_plan_direction'],z['direction_valid']):labels[group,int(frame)]=(y,m)
    reports=[];offsets=np.array([0,3,7,15,31])
    for variant in protocol['contract']['modes']:
        with np.load(a.run/f'{variant}_inner_held.npz') as z:pred,base,ids=z['predicted'],z['base'],z['indices']
        planned=[];target=[];masks=[];meta=[]
        for i,global_i in enumerate(ids):
            row=rows[global_i];y,m=labels[row['group'],row['frame']]
            planned.append(cartesian_residual(base[i,offsets],pred[i,offsets],configs[row['group']]));target.append(y);masks.append(m);meta.append(row)
        planned,target,masks=np.array(planned),np.array(target),np.array(masks)
        for task in sorted({r['task'] for r in meta}):
            for kind,cols,scale,floor in [('translation_m',[[0,1,2],[7,8,9]],.01,.0005),('rotation_rad',[[3,4,5],[10,11,12]],.1,.005)]:
                scene_metrics=[]
                for group in sorted({r['group'] for r in meta if r['task']==task}):
                    select=np.array([r['group']==group for r in meta]);actual=np.concatenate([planned[select][:,:,c]*scale for c in cols],axis=1);desired=np.concatenate([target[select][:,:,c]*scale for c in cols],axis=1);valid=np.concatenate([masks[select]]*2,axis=1)
                    error=np.mean((actual-desired)**2,axis=-1);zero=np.mean(desired**2,axis=-1);norm=np.linalg.norm(desired,axis=-1);pn=np.linalg.norm(actual,axis=-1);directions=valid&(norm>floor)
                    cosine=(actual*desired).sum(-1)/np.maximum(norm*pn,1e-15)
                    scene_metrics.append(dict(group=group,mse=float(error[valid].mean()),zero_mse=float(zero[valid].mean()),direction_slots=int(directions.sum()),positive_slots=int(((cosine>0)&directions).sum()),mean_cosine=float(cosine[directions].mean()) if directions.any() else None))
                mse=np.mean([r['mse'] for r in scene_metrics]);zero=np.mean([r['zero_mse'] for r in scene_metrics]);cosines=[r['mean_cosine'] for r in scene_metrics if r['mean_cosine'] is not None]
                reports.append(dict(variant=variant,task=task,kind=kind,scenes=scene_metrics,improvement_vs_zero=float(1-mse/zero) if zero>1e-16 else None,mean_scene_cosine=float(np.mean(cosines)) if cosines else None,direction_floor=floor))
    write_json(a.run/'direction_audit.json',dict(status='complete',reports=reports,scope='Same inner-held expert commands: compare FK(corrected command)-FK(base command) to FK(expert command)-FK(base command), including both arms and five fixed endpoints. Grounded robot kinematics only; not measured object motion or recovery/closed-loop success. Nontrivial direction thresholds descriptive only; zero/nearzero targets retained in MSE.'))
    for r in reports:
        if r['variant'] in ('semantic_bc','semantic_direction'):print(r['variant'],r['task'],r['kind'],r['improvement_vs_zero'],r['mean_scene_cosine'])

if __name__=='__main__':main()
