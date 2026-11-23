"""Frozen shared empirical gate on actual legal history; one opportunity only."""
from pathlib import Path
import numpy as np
from .output_support_policy import OutputSupportPolicy
from .causal_arm_gate import OneShotClosingGate
from .phase_target_policy import capture_all
from .empirical_onset_gate import GATES,load_frozen_gate,EmpiricalOnsetGate,candidate_packet,selected_actions
from .output_support_adapter import validate_native_arms
from .eval_diagnostics import array_hash
from .recovery_contract import rng_capture,rng_fingerprint


class SharedOnsetPolicy(OutputSupportPolicy):
    def __init__(self,base,training_run,condition,gate_run):
        model,self.gate_sha256=load_frozen_gate(gate_run,condition)
        self.selector=EmpiricalOnsetGate(model);self.detector=OneShotClosingGate()
        super().__init__(base,training_run,'A_bounded_clipped',force_zero=True)
        self.condition=condition
    def reset_history(self):
        super().reset_history();self.detector.reset();self.selector.reset()
    def predict(self,observation,instruction,noise,*,enabled):
        if enabled is not True:raise ValueError('Frozen candidate evaluated in every condition')
        query,previous=self.query_count,self.previous
        _,_,base=super().predict(observation,instruction,noise,enabled=True)
        before=rng_fingerprint(rng_capture())
        opportunity=self.detector.step(query,self.last_numeric['command_state'],base)
        capture_all(self,previous,base)
        window=opportunity['window'];raw=self.last_numeric['raw_delta']
        packet=candidate_packet(self.last_legal_inputs,raw) if window is not None and query==window[0] else None
        choice=self.selector.step(query,window,packet)
        actions,delta,prescription=selected_actions(base,raw,query,window,choice)
        after=rng_fingerprint(rng_capture())
        if before!=after or before!=self.last_diagnostics['rng_after_sampler']:raise RuntimeError('Gate modified sampler RNG')
        self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
        self.last_diagnostics.update(gate_decision=opportunity,empirical_gate=choice,gate_checkpoint_sha256=self.gate_sha256,
            gate_input_sha256=None if packet is None else {k:array_hash(v) for k,v in packet.items()},
            rng_after_gate=after,prescription=prescription,window_active=prescription['active'],applied_window=prescription['window'],
            arm_strength=prescription['strength'],gripper_strength=0.,action_chunk_sha256=array_hash(actions),
            applied_delta_sha256=array_hash(delta),max_abs_correction=float(np.abs(delta).max()),
            applied_max_abs_correction=float(np.abs(delta).max()),robot_domain=validate_native_arms(actions,self.joint_limits),
            actual_timing='Frozen empirical score at sole current-observation opportunity; reject consumes opportunity')
        return actions,delta,base
