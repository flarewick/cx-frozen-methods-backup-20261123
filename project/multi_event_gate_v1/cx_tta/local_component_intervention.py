"""Frozen A, fixed local window, independently masked arm/gripper residuals."""
import numpy as np
from .output_support_policy import OutputSupportPolicy
from .output_support_adapter import validate_native_arms
from .structured_action_adapter import ARM_INDICES
from .structured_features import pack_observable
from .eval_diagnostics import array_hash

FACTORS={'C00_zero':(0,0),'C10_arm_only':(1,0),
         'C01_gripper_only':(0,1),'C11_both':(1,1)}
CONDITIONS=tuple(FACTORS)
GRIPPERS=(6,13)


def strengths(condition,query,window):
    if condition not in FACTORS or type(query) is not int or query<0:
        raise ValueError('Registered condition and nonnegative query required')
    if (len(window)!=2 or any(type(v) is not int for v in window) or
            window[0]<0 or window[1]!=window[0]+2):
        raise ValueError('Fixed two-query half-open window required')
    within=window[0]<=query<window[1]
    arm,grip=FACTORS[condition] if within else (0,0)
    return arm,grip,within


def component_actions(base,raw,condition,query,window):
    base,raw=np.asarray(base),np.asarray(raw)
    if (base.shape!=(32,14) or raw.shape!=base.shape or
            base.dtype!=np.float32 or raw.dtype!=np.float32 or
            not np.isfinite(base).all() or not np.isfinite(raw).all()):
        raise ValueError('Finite float32 native32x14 base/proposal required')
    arm,grip,within=strengths(condition,query,window)
    delta=np.zeros_like(base);actions=base.copy()
    # Preserve disabled command columns exactly, including signed zero.
    for value,columns in ((arm,ARM_INDICES),(grip,GRIPPERS)):
        if value:
            delta[:,columns]=raw[:,columns]
            actions[:,columns]=base[:,columns]+raw[:,columns]
    return actions,delta,arm,grip,within


class LocalComponentPolicy(OutputSupportPolicy):
    def __init__(self,base,training_run,condition,window):
        if condition not in FACTORS:raise ValueError('Unknown component cell')
        strengths(condition,0,window)
        super().__init__(base,training_run,'A_bounded_clipped',force_zero=True)
        self.condition=condition;self.window=list(window)
        self.capture_queries={0,1,window[0],window[1]-1,window[1],window[1]+1}

    def predict(self,observation,instruction,noise,*,enabled):
        # All cells measure the same proposal, even C00; only applied columns
        # change. The frozen parent's force_zero isolates assembly below.
        if enabled is not True:raise ValueError('Common proposal inference required')
        query=self.query_count;previous=self.previous
        before=self.derived_checkpoint_sha256
        _,_,base=super().predict(observation,instruction,noise,enabled=True)
        raw=self.last_numeric['raw_delta'].copy()
        actions,delta,arm,grip,within=component_actions(base,raw,self.condition,query,self.window)
        domain=validate_native_arms(actions,self.joint_limits)
        if not (arm or grip) and actions.tobytes()!=base.tobytes():
            raise RuntimeError('Disabled intervention changed base bytes')
        if self.derived_checkpoint_sha256!=before:raise RuntimeError('Frozen checkpoint identity changed')
        self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
        self.last_diagnostics.update(within_window=within,window_active=bool(arm or grip),window=list(self.window),
            arm_strength=float(arm),gripper_strength=float(grip),component_factors=list(FACTORS[self.condition]),
            action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),
            max_abs_correction=float(np.abs(delta).max()),applied_max_abs_correction=float(np.abs(delta).max()),
            robot_domain=domain,intervention_scope='Fixed local window; both arms and both grippers receive the same respective factor',
            raw_proposal_scope='Same frozenA prediction in every cell,including00; not an applied00 correction')
        if query in self.capture_queries:
            joint,state=self.previous
            old_joint,old_state=previous if previous is not None else (np.zeros_like(joint),np.zeros_like(state))
            fields=pack_observable(joint[None],state[None],old_joint[None],old_state[None],
                np.asarray([previous is not None]),self.projection,self.robots)
            if any(array_hash(v)!=self.last_diagnostics['structured_input_sha256'][k] for k,v in fields.items()):
                raise RuntimeError('Actual component legal packet differs')
            self.last_legal_inputs={k:v.copy() for k,v in fields.items()};self.last_legal_inputs['base']=base[None].copy()
        else:self.last_legal_inputs=None
        return actions,delta,base
