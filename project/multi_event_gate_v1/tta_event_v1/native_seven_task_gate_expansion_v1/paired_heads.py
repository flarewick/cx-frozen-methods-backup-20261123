"""Frozen gate candidate plus the original qualified TTA auxiliary readout.

Both heads consume the same already-sampled base and legal observation fields.
The response model keeps its original hidden coordinates, ARM weight and rowspace.
No buffer transplant, extra VLA sampling or falsely relabelled qualification.
"""
import sys
import numpy as np
from .contract import W,GATE_TRAINING,plan

ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12],np.int64)

def build():
    import torch
    from cx_tta_feedback_confirm5.query import build_head
    sys.path.insert(0,str(W/'task_adapter_v1'))
    from task_adapters.adapter import StackingAdapter,read_checkpoint
    from candidate_eval_checkpoint import candidate
    from cx_tta.robot_kinematics import RobotFK
    from pathlib import Path
    import json
    p=plan()
    initial=read_checkpoint(GATE_TRAINING/'adapter_seed0.pt');meta=initial['metadata']
    gate=StackingAdapter(32,meta['geometry_mean']);gate.load_state_dict(initial['model'],strict=True)
    payload,checkpoint,spec=candidate('M_shared250_a050');assert spec['alpha']==.5
    frozen={k:v.clone()for k,v in gate.state_dict().items()if k.startswith('geometry_head.')or k in('caps','reg_mean','time_encoding')}
    gate.load_state_dict(payload['model'],strict=True);gate.eval().requires_grad_(False)
    assert all(torch.equal(gate.state_dict()[k],v)for k,v in frozen.items())
    aux,auxrobots=build_head(p['plain_training'])
    robot=json.loads(Path(meta['robot_config_path']).read_text());robots=[RobotFK(robot,side)for side in('left','right')]
    return gate,aux,robots,auxrobots

def forward(head,legal,base):
    import torch
    from cx_tta.structured_action_adapter import FORWARD_KEYS
    if set(legal)!=FORWARD_KEYS:raise ValueError('Exact original legal fields required')
    if base.dtype!=np.float32 or base.shape!=(32,14):raise ValueError('Original once-sampled base required')
    captured={}
    def hook(name):
        def record(module,args,value):captured[name]=value.detach().cpu().numpy().copy()[0]
        return record
    hooks=[head.action_head[1].register_forward_hook(hook('hidden')),head.action_head[2].register_forward_hook(hook('logits'))]
    try:
        with torch.inference_mode():
            result=head({k:torch.from_numpy(v.copy())for k,v in legal.items()},torch.from_numpy(base[None].copy()),enabled=True)
        raw=result['delta'][0].cpu().numpy().copy()
    finally:
        for h in hooks:h.remove()
    assert raw.shape==(32,14)and not np.any(raw[:,[6,13]])and np.max(np.abs(raw))<=.100001
    return dict(raw_delta=raw,**captured)

def combine(reference,auxiliary_reference,auxiliary_direction):
    if reference.shape!=(32,14)or reference.dtype!=np.float32:raise ValueError('Frozen gate raw candidate required')
    if auxiliary_direction is None:return None,dict(changed=False,reason='no_auxiliary_direction')
    ref=np.asarray(auxiliary_reference)[:,ARM];direction=np.asarray(auxiliary_direction)
    if direction.shape!=(32,12)or direction.dtype!=np.float32 or not np.isfinite(direction).all():raise ValueError('Qualified auxiliary ARM direction required')
    shift=direction-ref
    if not np.isfinite(shift).all()or np.max(np.abs(shift))>.020001:raise ValueError('Original auxiliary trust shift exceeded')
    original=reference[:,ARM]
    if not np.any(shift):return original.copy(),dict(changed=False,reason='exact_zero_auxiliary_fast_increment')
    candidate=np.clip(original+shift,np.float32(-.1),np.float32(.1)).astype(np.float32,copy=False)
    a,b=original.astype(np.float64).ravel(),candidate.astype(np.float64).ravel();na,nb=np.linalg.norm(a),np.linalg.norm(b)
    cosine=1. if na==0 and nb==0 else 0. if na==0 or nb==0 else float(a@b/(na*nb))
    if cosine<.8:return original.copy(),dict(changed=False,reason='gate_candidate_direction_trust_rejected',cosine=cosine)
    return candidate,dict(changed=not np.array_equal(original,candidate),reason='qualified_auxiliary_fast_increment_added_to_gate_candidate',cosine=cosine,max_raw_shift=float(np.max(np.abs(candidate-original))),alpha=.5)
