"""Explicitly scaled, validated offline candidates on the unchanged event policy."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from task_adapters.policy import StackingPolicy as OriginalPolicy
from cx_tta.multi_event import EventConfig
from cx_tta.structured_action_adapter import ARM_INDICES
from cx_tta.eval_diagnostics import array_hash
from cx_tta.official_onset_domain import observe_native_arms
from candidate_eval_reference import FILE as REFERENCE_FILE,load as load_reference,verify_query,sha
from candidate_eval_checkpoint import CANDIDATES,candidate,STUDY
REFERENCE=None

def scaled_actions(base,delta,alpha):
    if base.shape!=(32,14) or delta.shape!=base.shape or base.dtype!=np.float32 or delta.dtype!=np.float32 or not np.isfinite(base).all() or not np.isfinite(delta).all() or not 0<=alpha<=1:raise ValueError('Finite native float32 chunks and bounded scale required')
    if np.abs(delta[:,ARM_INDICES]).max()>.100001 or delta[:,[6,13]].any():raise ValueError('Original ARM/GRIP contract violated')
    applied=np.zeros_like(base);applied[:,ARM_INDICES]=delta[:,ARM_INDICES]*np.float32(alpha)
    actions=base.copy();mask=applied!=0;actions[mask]=base[mask]+applied[mask]
    return actions,applied

class CandidatePolicy(OriginalPolicy):
    def __init__(self,base,training_run,condition,config=EventConfig()):
        external=condition
        super().__init__(base,training_run,'M_event_full' if external in CANDIDATES else condition,config)
        self.candidate_condition=external;self.candidate_alpha=1.;self.reference_intervened=False
        if external in CANDIDATES:
            payload,path,spec=candidate(external)
            original={k:v.clone() for k,v in self.model.state_dict().items() if k.startswith('geometry_head.') or k in ('caps','reg_mean','time_encoding')}
            self.model.load_state_dict(payload['model'],strict=True);self.model.eval().requires_grad_(False)
            for key,value in original.items():
                if not torch.equal(self.model.state_dict()[key],value):raise RuntimeError('Frozen candidate component differs: '+key)
            self.candidate_alpha=spec['alpha'];self.derived_checkpoint_sha256=sha(path)
            self.metadata=dict(self.metadata,candidate_provenance=copy.deepcopy(payload['metadata']),training_tasks=payload['metadata']['job']['tasks'])
        self.condition=external
    def reset_history(self):
        super().reset_history();self.reference_intervened=False
    def predict(self,observation,instruction,noise,*,enabled):
        external=self.condition
        if external not in CANDIDATES:return super().predict(observation,instruction,noise,enabled=enabled)
        if not enabled or self.query_recorder is not None:raise RuntimeError('Explicit candidate activation and canonical trace required')
        self.condition='M_event_full'
        try:_,delta,base=super().predict(observation,instruction,noise,enabled=enabled)
        finally:self.condition=external
        global REFERENCE
        if REFERENCE is None:REFERENCE=load_reference()
        checked=verify_query(REFERENCE,self.episode_seed,instruction,self.last_diagnostics,self.reference_intervened)
        actions,delta=scaled_actions(base,delta,self.candidate_alpha)
        active=self.last_diagnostics['multi_event']['active']
        self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
        self.last_diagnostics.update(evaluated_condition=external,candidate_alpha=self.candidate_alpha,
            candidate_checkpoint_sha256=self.derived_checkpoint_sha256,arm_strength=self.candidate_alpha if active else 0.,
            action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),max_abs_correction=float(np.abs(delta).max()),
            applied_max_abs_correction=float(np.abs(delta).max()),robot_domain=observe_native_arms(actions,self.joint_limits),
            reference_baseline_prefix_checked=checked,reference_assertion_uses_no_policy_outcomes=True)
        self.last_diagnostics['prescription']=dict(self.last_diagnostics['prescription'],strength=self.candidate_alpha if active else 0.)
        if active:self.reference_intervened=True
        return actions,delta,base
