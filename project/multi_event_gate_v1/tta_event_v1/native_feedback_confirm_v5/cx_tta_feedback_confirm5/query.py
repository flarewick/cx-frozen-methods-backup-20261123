"""Frozen legal-input discovery and finite signed candidate controls."""
import numpy as np

ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12])


def compose(base,raw,branch):
    from .contract import BRANCHES
    if branch not in BRANCHES:raise ValueError('Unknown fixed treatment')
    if any(v.shape!=(32,14) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in (base,raw)):
        raise ValueError('Native finite float32 chunks required')
    if np.any(raw[:,[6,13]]) or np.max(np.abs(raw[:,ARM]))>.100001:raise ValueError('Original residual limits changed')
    d=np.zeros_like(raw)
    if branch=='C_original':d[:,ARM]=raw[:,ARM]
    elif branch=='C_negative':d[:,ARM]=-raw[:,ARM]
    elif branch=='C_permuted':
        for start in (0,7):d[:,start:start+6]=np.roll(raw[:,start:start+6],1,axis=1)
    actions=base.copy();mask=d!=0;actions[mask]=base[mask]+d[mask]
    if not np.array_equal(actions[:,[6,13]],base[:,[6,13]]):raise ValueError('Treatment changed planned gripper')
    return actions


def build_head(training):
    import json,torch
    from pathlib import Path
    from cx_tta.stacking_adapter import read_checkpoint,StackingAdapter
    from cx_tta.robot_kinematics import RobotFK
    payload=read_checkpoint(Path(training)/'adapter_seed0.pt');meta=payload['metadata']
    model=StackingAdapter(32,meta['geometry_mean']);model.load_state_dict(payload['model'],strict=True)
    model.eval().requires_grad_(False)
    robot=json.loads(Path(meta['robot_config_path']).read_text())
    return model,[RobotFK(robot,side) for side in ('left','right')]


def query(base,head,robots,projection,initial,images,instruction,previous):
    """No task/seed/physics/expert/goal labels. One internal-noise action query."""
    import torch
    from cx_tta.structured_features import pack_observable
    from cx_tta.structured_action_adapter import FORWARD_KEYS
    state=initial['command_state'].copy();joint=initial['joint'][0].copy()
    previous_joint,previous_state=previous if previous is not None else (np.zeros_like(joint),np.zeros_like(state))
    x=pack_observable(joint[None],state[None],previous_joint[None],previous_state[None],
        np.asarray([previous is not None]),projection,robots)
    if set(x)!=FORWARD_KEYS:raise ValueError('Changed legal input schema')
    policy_obs=dict(state=state.copy(),prompt=instruction,images={k:v.transpose(2,0,1).copy()
        for k,v in zip(('cam_high','cam_right_wrist','cam_left_wrist'),images)})
    # The original eager sample_actions body, no torch.compile/old actor parity claim.
    actions=np.asarray(base.infer(policy_obs,noise=None)['actions'],np.float32).copy()
    if actions.shape!=(32,14) or not np.isfinite(actions).all():raise ValueError('Bad once-sampled base')
    with torch.inference_mode():
        result=head({k:torch.from_numpy(v.copy()) for k,v in x.items()},torch.from_numpy(actions[None]),enabled=True)
    raw=result['delta'][0].cpu().numpy().copy()
    compose(actions,raw,'C_original')
    return dict(base=actions,raw_delta=raw,legal={**x,'base':actions[None].copy()},previous=(joint,state))
