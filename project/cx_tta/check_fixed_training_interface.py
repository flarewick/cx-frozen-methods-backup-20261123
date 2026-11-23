"""One CPU forward/backward contract check; no optimizer or parameter update."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .paths import write_json
from .fixed_recovery_dataset import load_fixed_cache,verify_scene_disjoint
from .cache_recovery import read
from .train_scene_crossfit import candidate_loss
from .time_encoding import TimeEncodedAdapter
from .pose_adapter import TorchFK
from .robot_kinematics import RobotFK
from .train_rotation_curriculum import state_hash

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    inputs,labels,rows=load_fixed_cache(run,scope='developer_same_task_scene_holdout');p=read(run/'protocol.json')
    source=Path(read(Path(p['rows'][0]['source'])/'manifest.json')['source']);config=read(source/'resolved_config.json')
    robots=[TorchFK(RobotFK(config,s)).float() for s in ('left','right')]
    x=torch.from_numpy(inputs['semantic']);base=torch.from_numpy(inputs['base']);target=torch.from_numpy(labels['target']);valid=torch.from_numpy(labels['valid'])
    pairs=torch.tensor([[j,31,i] for i,j in enumerate(p['history_indices']) if j>=0],dtype=torch.long).reshape(-1,3)
    for before,last,after in pairs.tolist():
        if rows[before]['branch_group']!=rows[after]['branch_group'] or rows[before]['group']!=rows[after]['group'] or not valid[before,last]:raise ValueError('Invalid teacher seam')
    weights=torch.ones(len(rows))/len(rows);pw=torch.ones(len(pairs))/max(1,len(pairs));reports=[]
    for encoding in ('repeat','fourier'):
        torch.manual_seed(19021);model=TimeEncodedAdapter(157,encoding=encoding,envelope=True);initial=state_hash(model)
        def evaluate(t):
            model.zero_grad();pred=model(x,base);loss=candidate_loss(pred,t,valid,weights,pairs,pw,robots,model.caps);loss.backward()
            grad=torch.cat([v.grad.flatten() for v in model.parameters() if v.grad is not None]);return pred.detach(),loss.detach(),grad.detach().clone()
        pred,loss,grad=evaluate(target);perturbed=target.clone();perturbed[~valid]=1000
        _,other,changed=evaluate(perturbed)
        if not torch.equal(pred,base) or not torch.equal(loss,other) or not torch.equal(grad,changed):raise ValueError('Init/masked-label contract failed')
        if not torch.isfinite(grad).all() or grad.norm()==0 or state_hash(model)!=initial:raise ValueError('Gradient/unchanged parameter contract failed')
        reports.append(dict(encoding=encoding,zero_init_exact=True,invalid_tail_loss_gradient_exact=True,parameters_unchanged=True,finite_nonzero_gradient=True,loss=float(loss),parameters=sum(v.numel() for v in model.parameters())))
    scenes=sorted({r['group'] for r in rows})
    for held in scenes:verify_scene_disjoint([r for r in rows if r['group']!=held],[r for r in rows if r['group']==held])
    write_json(run/'training_interface_check.json',dict(passed=True,optimizer_steps=0,rows=len(rows),scene_groups=scenes,seam_pairs=len(pairs),
        forward_keys=['semantic_context','same_state_base'],label_keys=sorted(labels),reports=reports,
        source_sha256={str(f):digest(f) for f in [run/'audit.json',run/'data.npz',Path(__file__),Path(__file__).with_name('fixed_recovery_dataset.py'),
            Path(__file__).with_name('train_scene_crossfit.py'),Path(__file__).with_name('time_encoding.py'),Path(__file__).with_name('recovery_temporal.py')]},
        scope='Existing training objective and reader interface only; no fit, new model or task success claim'))
    print('TRAINING_INTERFACE_PASS',len(rows),'rows',len(scenes),'scenes; optimizer_steps=0',flush=True)

if __name__=='__main__':main()
