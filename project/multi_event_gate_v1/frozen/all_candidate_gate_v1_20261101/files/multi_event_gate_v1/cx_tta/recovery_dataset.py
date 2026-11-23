"""Audited recovery-fit data; observable forward inputs separated from labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import write_json

SCOPE='developer_recovery_fit_interface_only'

def load_recovery_cache(run, *, scope):
    """Require an explicit developer-fit scope; never infer a held-out split."""
    run=Path(run).resolve()
    if scope!=SCOPE:raise ValueError('This small developer cache has no formal train/test approval')
    read=lambda p:json.loads(p.read_text())
    audit=read(run/'audit.json');protocol=read(run/'protocol.json')
    if not audit['passed'] or audit['approved_scope']!=scope:raise ValueError('Audited scope required')
    for f,sha in audit['source_sha256'].items():
        if digest(Path(f))!=sha:raise ValueError('Changed audited artifact '+f)
    # Labels and diagnostic metadata cannot accidentally become extra forward channels.
    with np.load(run/'data.npz') as z:
        inputs={k:z[k].copy() for k in ('context','base')}
        labels={k:z[k].copy() for k in ('target','valid')}
    metadata=[dict(group=r['group'],task=r['task'],native_count=r['native_count'],
                   native_limit=r['native_limit'],scope=scope) for r in protocol['rows']]
    return inputs,labels,metadata

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();inputs,labels,rows=load_recovery_cache(run,scope=SCOPE)
    import torch
    from .pose_adapter import PoseAdapter,TorchFK
    from .robot_kinematics import RobotFK
    from .train_pose_capacity import OFFSETS,fk_endpoints
    torch.set_num_threads(2);torch.manual_seed(0)
    model=PoseAdapter(inputs['context'].shape[1],proposal=True)
    x=torch.from_numpy(inputs['context']);base=torch.from_numpy(inputs['base'])
    target=torch.from_numpy(labels['target']);valid=torch.from_numpy(labels['valid'])
    protocol=json.loads((run/'protocol.json').read_text())
    prepared=json.loads((Path(protocol['preprocess']).parent/'manifest.json').read_text())
    config=json.loads(Path(prepared['config']).read_text())
    robots=[TorchFK(RobotFK(config,s)).float() for s in ('left','right')]
    def loss_and_grad(t):
        model.zero_grad();pred=model(x,base)
        bc=((((pred-t)/model.caps).square().mean(-1)*valid).sum(-1)/valid.sum(-1)).mean()
        pe=fk_endpoints(pred,robots);te=fk_endpoints(t,robots)
        mask=valid[:,OFFSETS,None].expand(-1,-1,2)
        chord=(pe[...,:3,:3]-te[...,:3,:3]).square().sum((-2,-1))/2
        pose=((chord/.01*mask).sum((1,2))/mask.sum((1,2)).clamp(min=1)).mean()
        loss=bc+pose
        loss.backward();grad=torch.cat([p.grad.flatten() for p in model.parameters()])
        return pred.detach(),loss.detach(),grad.detach().clone()
    pred,loss,grad=loss_and_grad(target)
    changed=target.clone();changed[~valid]=1000
    _,padded_loss,padded_grad=loss_and_grad(changed)
    if not torch.equal(pred,base):raise ValueError('Zero proposal changed base')
    if not torch.equal(loss,padded_loss) or not torch.equal(grad,padded_grad):raise ValueError('Invalid terminal targets affect BC/SO(3) training')
    if not torch.isfinite(grad).all() or grad.norm()==0:raise ValueError('No finite learning signal')
    write_json(run/'training_interface_check.json',dict(passed=True,scope=SCOPE,rows=len(rows),
        forward_keys=sorted(inputs),label_keys=sorted(labels),context_shape=list(x.shape),base_shape=list(base.shape),
        valid_counts=valid.sum(-1).tolist(),zero_init_exact=True,invalid_tail_loss_and_gradient_invariant=True,
        finite_nonzero_gradient=True,bc_plus_so3_loss=float(loss),parameters=sum(p.numel() for p in model.parameters()),
        source_sha256={str(p):digest(p) for p in [run/'audit.json',Path(__file__),Path(__file__).with_name('pose_adapter.py'),
            Path(__file__).with_name('train_pose_capacity.py'),Path(prepared['config'])]},
        limitation='Loader and existing BC+SO(3) gradient/mask check only; no optimizer update or recovery efficacy result.'))
    print(json.dumps(dict(passed=True,rows=len(rows),valid_counts=valid.sum(-1).tolist(),forward_keys=sorted(inputs)),indent=2))

if __name__=='__main__':main()
