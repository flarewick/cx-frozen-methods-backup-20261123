"""Post-training fit-state diagnostic; no new optimization or model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .audit_pose_capacity import numpy_fk,angles,episode_metrics
from .paths import write_json
from .pose_adapter import PoseAdapter
from .robot_kinematics import RobotFK

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();read=lambda f:json.loads(f.read_text());protocol=read(run/'protocol.json')
    if not read(run/'audit.json')['passed']:raise ValueError('Capacity audit required')
    torch.set_num_threads(2);rows=protocol['rows'];prepared=Path(protocol['prepared']);config=read(Path(protocol['config']));robots=[RobotFK(config,s) for s in ('left','right')]
    with np.load(prepared/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    ids=np.array([i for i,r in enumerate(rows) if r['pilot_split']=='fit']);context=torch.from_numpy(data['context'][ids]);base=torch.from_numpy(data['base'][ids]);target=data['target'][ids];valid=data['valid'][ids]
    def endpoints(values):return np.stack([numpy_fk(r,values[:,protocol['offsets'],start:start+6].astype(float)) for r,start in zip(robots,(0,7))],2)
    bp,tp=endpoints(base.numpy()),endpoints(target);ba=angles(bp,tp);mask=np.broadcast_to(valid[:,protocol['offsets'],None],ba.shape);large=mask&(ba>=45.);reports=[];sources=[run/'protocol.json',run/'audit.json',prepared/'data.npz',Path(__file__)]
    for seed in protocol['seeds']:
        for variant in protocol['variants']:
            checkpoint=run/f'{variant}_s{seed}.pt';sources.append(checkpoint);model=PoseAdapter(context.shape[1],proposal=variant!='bounded_bc');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            with torch.no_grad():pred=np.concatenate([model(context[i:i+64],base[i:i+64]).numpy() for i in range(0,len(ids),64)])
            pp=endpoints(pred);ang=angles(pp,tp);tr=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1);metrics,episode_rows=episode_metrics(ang,ba,tr,mask,large,ids,rows,pred,target,valid)
            path=run/f'fit_geometry_{variant}_s{seed}.npz';np.savez_compressed(path,indices=ids,angle_deg=ang,base_angle_deg=ba,translation_m=tr,valid=mask,large=large);sources.append(path)
            reports.append(dict(seed=seed,variant=variant,metrics=metrics,episodes=episode_rows,max_arm_correction_rad=float(np.abs((pred-base.numpy())[...,[0,1,2,3,4,5,7,8,9,10,11,12]]).max())))
    write_json(run/'fit_capacity.json',dict(status='complete',reports=reports,source_sha256={str(f):digest(f) for f in sources},scope='Post hoc diagnostic of fixed final checkpoints on existing training rows. No additional optimization/selection. Fit error cannot be reported as held success or generalization.'))
    print(json.dumps([dict(seed=r['seed'],variant=r['variant'],large={t:dict(base=v['large_base_angle_deg'],predicted=v['large_angle_deg']) for t,v in r['metrics'].items() if v['large_angle_deg'] is not None},max_arm_correction_rad=r['max_arm_correction_rad']) for r in reports],indent=2))

if __name__=='__main__':main()
