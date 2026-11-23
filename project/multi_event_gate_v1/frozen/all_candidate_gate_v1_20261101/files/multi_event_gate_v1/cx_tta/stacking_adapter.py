"""Shared bounded expert-initialization head; current observation in train and deploy."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .structured_action_adapter import StructuredActionAdapter, ARM_INDICES
from .audit import digest
from .paths import output_path

SCOPE='stacking_shared_expert_initialization_v1'


class StackingAdapter(StructuredActionAdapter):
    def __init__(self,channels=32,geometry_mean=None):
        super().__init__(channels,geometry_mean)
        mean=self.reg_mean
        del self.reg_mean
        self.register_buffer('reg_mean',mean)

    def forward(self,x,base,enabled=True):
        self._validate(x,base)
        # Expert saved frames are not actual previous32 native policy actions.
        # The same current-only restriction is enforced in deployment.
        current={k:torch.zeros_like(v) if k.startswith('past_') or k=='has_history' else v for k,v in x.items()}
        out=super().forward(current,base,enabled)
        mask=torch.ones(14,device=base.device,dtype=base.dtype);mask[[6,13]]=0
        out['delta']=out['delta']*mask
        out['actions']=torch.where(out['delta']!=0,base+out['delta'],base)
        return out


def arm_loss(delta,base,target,valid,weights):
    if any(not torch.isfinite(v).all() for v in (delta,base,weights)) or valid.dtype!=torch.bool:
        raise ValueError('Finite actions/weights and Boolean validity required')
    if base.shape!=delta.shape or target.shape!=base.shape or base.shape[1:]!=(32,14):
        raise ValueError('Expected native [N,32,14] chunks')
    if valid.shape!=base.shape[:2] or weights.shape!=(len(base),) or (weights<0).any():
        raise ValueError('Invalid mask/weights')
    active=valid.bool() & (weights[:,None]>0)
    if not torch.isfinite(target[active]).all():raise ValueError('Nonfinite active expert target')
    safe=torch.where(active[...,None],target,base)
    desired=((safe-base)/.1).clamp(-1,1)[...,ARM_INDICES]
    per_step=F.smooth_l1_loss(delta[...,ARM_INDICES]/.1,desired,reduction='none').mean(-1)
    per_row=(per_step*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return (per_row*weights).sum()


def scene_weights(rows):
    from collections import Counter
    tasks=Counter(r['task'] for r in rows)
    groups=Counter((r['task'],r['seed']) for r in rows)
    scenes=Counter(task for task,seed in groups)
    return np.asarray([1/(len(tasks)*scenes[r['task']]*groups[(r['task'],r['seed'])]) for r in rows],np.float32)


def read_checkpoint(path,*,require_audit=True):
    path=output_path(Path(path));run=path.parent
    plan=json.loads((run/'training_plan.json').read_text())
    result=json.loads((run/'training_result.json').read_text())
    if path.name!='adapter_seed0.pt' or plan.get('scope')!=SCOPE or result.get('status')!='complete':
        raise ValueError('Complete frozen stacking training required')
    if result['checkpoint_sha256']!=digest(path):raise ValueError('Checkpoint changed')
    payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
    if (meta['training_plan_sha256']!=digest(run/'training_plan.json') or meta['scope']!=SCOPE
        or meta['updates']!=plan['updates'] or meta['seed']!=plan['seed']
        or meta['history_mode']!='current_only' or meta['gripper_residual']!=0
        or meta['arm_cap_rad']!=.1 or meta['cache_manifest_sha256']!=digest(run/'cache_manifest.json')):
        raise ValueError('Stacking checkpoint contract changed')
    for name in ('projection','robot_config'):
        if meta[name+'_path']!=plan[name] or digest(Path(meta[name+'_path']))!=meta[name+'_sha256']:
            raise ValueError('Changed '+name)
    if require_audit:
        audit=json.loads((run/'training_audit.json').read_text())
        if not audit['passed'] or audit['checkpoint_sha256']!=digest(path):raise ValueError('CPU reload audit required')
    model=StackingAdapter(32,meta['geometry_mean']);model.load_state_dict(payload['model'],strict=True)
    return payload
