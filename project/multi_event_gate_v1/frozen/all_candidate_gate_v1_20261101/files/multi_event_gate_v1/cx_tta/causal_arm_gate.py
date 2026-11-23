"""Latched current-state/current-base trigger, never retrospective closure."""
import numpy as np
from .eval_diagnostics import array_hash
from .output_support_policy import OutputSupportPolicy
from .output_support_adapter import validate_native_arms
from .local_component_intervention import component_actions
from .structured_features import pack_observable

CONDITIONS=('Z_zero','A_causal_arm')
SMOKE_CONDITIONS=(*CONDITIONS,'A_fixed_arm')
GRIPPERS=(6,13)


class OneShotClosingGate:
    def __init__(self):self.reset()
    def reset(self):self.last_query=-1;self.first_event=None;self.window=None
    def step(self,query,state,base):
        state,base=np.asarray(state),np.asarray(base)
        if type(query) is not int or query!=self.last_query+1:
            raise ValueError('Contiguous causal query order required')
        if state.shape!=(14,) or base.shape!=(32,14) or state.dtype!=np.float32 or base.dtype!=np.float32 or not np.isfinite(state).all() or not np.isfinite(base).all():
            raise ValueError('Finite native float32 current-state/base required')
        # Only event measurement is clipped; actual base is never projected.
        decrease=state[list(GRIPPERS)]-np.clip(base[:,GRIPPERS],0,1).min(axis=0)
        event_now=bool(np.any(decrease>=.1));new_event=self.first_event is None and event_now
        if new_event:self.first_event=query;self.window=[query+1,query+3]
        active=self.window is not None and self.window[0]<=query<self.window[1]
        self.last_query=query
        return dict(query=query,closing_decrease=decrease.tolist(),event_now=event_now,
            new_event=bool(new_event),first_event_query=self.first_event,
            window=None if self.window is None else self.window.copy(),active=bool(active),
            state_sha256=array_hash(state),base_sha256=array_hash(base),threshold=.1)


class CausalArmPolicy(OutputSupportPolicy):
    def __init__(self,base,training_run,condition,fixed_window=None):
        if condition not in SMOKE_CONDITIONS:raise ValueError('Unknown causal study condition')
        self.detector=OneShotClosingGate()
        super().__init__(base,training_run,'A_bounded_clipped',force_zero=True)
        self.condition=condition;self.fixed_window=fixed_window
        if condition=='A_fixed_arm' and fixed_window is None:raise ValueError('Fixed window is interface-only control')
    def reset_history(self):
        super().reset_history();self.detector.reset()
    def predict(self,observation,instruction,noise,*,enabled):
        if enabled is not True:raise ValueError('Common frozenA proposal evaluation required')
        query=self.query_count;previous=self.previous
        _,_,base=super().predict(observation,instruction,noise,enabled=True)
        state=self.last_numeric['command_state'];raw=self.last_numeric['raw_delta'].copy()
        decision=self.detector.step(query,state,base)
        chosen=self.fixed_window if self.condition=='A_fixed_arm' else decision['window']
        active=self.condition!='Z_zero' and chosen is not None and chosen[0]<=query<chosen[1]
        if active:actions,delta,_,_,_=component_actions(base,raw,'C10_arm_only',query,chosen)
        else:actions=base.copy();delta=np.zeros_like(base)
        domain=validate_native_arms(actions,self.joint_limits)
        if not active and actions.tobytes()!=base.tobytes():raise RuntimeError('Causal off is not current-base identity')
        self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
        self.last_diagnostics.update(gate_decision=decision,window_active=bool(active),applied_window=chosen,
            arm_strength=float(active),gripper_strength=0.,actual_timing='fixed_interface_control' if self.condition=='A_fixed_arm' else 'current_state_and_current_base_only',
            action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),
            max_abs_correction=float(np.abs(delta).max()),applied_max_abs_correction=float(np.abs(delta).max()),robot_domain=domain)
        capture={0,1}
        for w in (decision['window'],self.fixed_window):
            if w is not None:capture.update((w[0]-1,w[0],w[1]-1,w[1],w[1]+1))
        if query in capture:
            joint,now=self.previous;old_joint,old_state=previous if previous is not None else (np.zeros_like(joint),np.zeros_like(now))
            x=pack_observable(joint[None],now[None],old_joint[None],old_state[None],np.asarray([previous is not None]),self.projection,self.robots)
            if any(array_hash(v)!=self.last_diagnostics['structured_input_sha256'][k] for k,v in x.items()):raise RuntimeError('Actual causal legal packet differs')
            self.last_legal_inputs={k:v.copy() for k,v in x.items()};self.last_legal_inputs['base']=base[None].copy()
        else:self.last_legal_inputs=None
        return actions,delta,base
