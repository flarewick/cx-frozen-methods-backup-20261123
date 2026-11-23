"""Frozen-A temporal intervention: identity outside a registered short window."""
import numpy as np
from .output_support_policy import OutputSupportPolicy
from .structured_features import pack_observable
from .eval_diagnostics import array_hash

CONDITIONS=('L0','A_always','A_close_window','A_postclose_window')


def choose_windows(rows):
    """Offline diagnostic landmarks, not a learned deployable task router."""
    closure=[]
    for i,row in enumerate(rows):
        before,after=row['native_before']['control'],row['native_after']['control']
        if any(before[s+'_command'][-1]>=.5 and after[s+'_command'][-1]<.5 for s in ('left','right')):
            closure.append(i)
    if not closure:raise ValueError('No source closure landmark; no substitute scene/window')
    k=closure[0]
    return dict(first_closure_query=k,A_close_window=[max(0,k-1),max(0,k-1)+2],
                A_postclose_window=[k+1,k+3],definition='Half-open query intervals, exactly2 chunks<=64 native actions; original total budget unchanged')


def active_at(condition,query,windows):
    if condition not in CONDITIONS or type(query) is not int or query<0:raise ValueError('Invalid schedule')
    if condition=='L0':return False
    if condition=='A_always':return True
    start,end=windows[condition]
    if type(start) is not int or type(end) is not int or start<0 or end!=start+2:raise ValueError('Registered2-chunk window required')
    return start<=query<end


class LocalInterventionPolicy(OutputSupportPolicy):
    def __init__(self,base,training_run,condition):
        super().__init__(base,training_run,'L0' if condition=='L0' else 'A_bounded_clipped')
        if condition not in CONDITIONS:raise ValueError('Local condition')
        self.condition=condition;self.windows=None;self.capture_queries=set()

    def set_windows(self,windows):
        self.windows=windows
        capture={0,1}
        for name in CONDITIONS[2:]:
            start,end=windows[name];capture.update((start,end-1,end))
        self.capture_queries=capture

    def predict(self,observation,instruction,noise,*,enabled):
        if self.windows is None:raise ValueError('Sealed schedule missing')
        query=self.query_count;previous=self.previous
        active=active_at(self.condition,query,self.windows)
        self.force_zero=not active
        actions,delta,base=super().predict(observation,instruction,noise,enabled=enabled)
        if not active and (actions.tobytes()!=base.tobytes() or np.any(delta)):
            raise RuntimeError('Window-off is not exact L0 identity')
        diagnostic=self.last_diagnostics
        diagnostic.update(window_active=active,arm_strength=float(active),gripper_strength=float(active),
                          window_schedule=self.windows,intervention_scope='FrozenA, time mask only; no task-specific model/strength')
        if query in self.capture_queries:
            current_joint,current_state=self.previous
            old_joint,old_state=previous if previous is not None else (np.zeros_like(current_joint),np.zeros_like(current_state))
            packed=pack_observable(current_joint[None],current_state[None],old_joint[None],old_state[None],
                np.asarray([previous is not None]),self.projection,self.robots)
            if any(array_hash(v)!=diagnostic['structured_input_sha256'][k] for k,v in packed.items()):
                raise RuntimeError('Reconstructed actual legal packet differs')
            self.last_legal_inputs={k:v.copy() for k,v in packed.items()};self.last_legal_inputs['base']=base[None].copy()
        else:self.last_legal_inputs=None
        return actions,delta,base
