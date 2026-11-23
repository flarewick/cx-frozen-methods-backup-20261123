"""Live multi-event wrapper over a copied frozen candidate/sampling path."""
import numpy as np
from .official_onset_domain import OfficialOutputSupportPolicy, observe_native_arms
from .phase_target_policy import capture_all
from .empirical_onset_gate import candidate_packet,load_frozen_gate
from .multi_event import EventConfig,EventDetector,LocalController,apply_arm_candidate
from .eval_diagnostics import array_hash
from .recovery_contract import rng_capture,rng_fingerprint

CONDITIONS=('M_shadow','M_first_full','M_event_full','M_legacy_candidate','M_legacy_observation')


class MultiEventPolicy(OfficialOutputSupportPolicy):
    def __init__(self,base,training_run,condition,gate_run,config=EventConfig()):
        if condition not in CONDITIONS:raise ValueError('Unknown multi-event condition')
        if condition=='M_first_full':config=EventConfig(**dict(config.to_dict(),max_interventions=1))
        self.event_config=config
        self.detector=EventDetector(config);self.controller=LocalController(config)
        # Old model is a diagnostic transfer baseline, not a multi-event-trained gate.
        self.selector_model=None;self.gate_sha256=None
        if condition in ('M_shadow','M_legacy_candidate','M_legacy_observation'):
            name='G_observation_cost2' if condition=='M_legacy_observation' else 'G_candidate_cost2'
            self.selector_model,self.gate_sha256=load_frozen_gate(gate_run,name)
        super().__init__(base,training_run,'A_bounded_clipped',force_zero=True)
        self.condition=condition

    def reset_history(self):
        super().reset_history();self.detector.reset();self.controller.reset();self.last_event_packet=None

    def predict(self,observation,instruction,noise,*,enabled):
        if enabled is not True:raise ValueError('Common candidate inference required')
        query,previous=self.query_count,self.previous
        _,_,base=super().predict(observation,instruction,noise,enabled=True)
        before=rng_fingerprint(rng_capture())
        capture_all(self,previous,base)
        raw=self.last_numeric['raw_delta'];packet=candidate_packet(self.last_legal_inputs,raw)
        event,evidence=self.detector.step(query,packet)
        def score(p,context):
            if self.selector_model is not None:return self.selector_model.predict(p)['class_scores']
            # Explicit always-admit intervention ablation, never a probability claim.
            return dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
        decision=self.controller.step(query,event,packet,score,enabled=self.condition!='M_shadow')
        actions,delta=apply_arm_candidate(base,raw,decision['active'])
        after=rng_fingerprint(rng_capture())
        if before!=after or before!=self.last_diagnostics['rng_after_sampler']:
            raise RuntimeError('Event/selector changed VLA sampler RNG')
        contexts=[r['context'] for r in decision['records'] if 'context' in r]
        self.last_event_packet=None if not contexts else dict(packet,event_context=np.asarray(contexts,np.float32))
        self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
        self.last_diagnostics.update(multi_event=dict(event=event,evidence=evidence,**decision),
            gate_checkpoint_sha256=self.gate_sha256,gate_training_scope='old_first_opportunity_only' if self.selector_model is not None else 'no_learned_selector',
            scoring_scope='diagnostic_uncalibrated_transfer' if self.selector_model is not None else 'always_admit_ablation',
            rng_after_gate=after,condition=self.condition,
            prescription=dict(active=decision['active'],strength=float(decision['active']),window=[query,query+1]),
            arm_strength=float(decision['active']),gripper_strength=0.,
            action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),
            max_abs_correction=float(np.abs(delta).max()),applied_max_abs_correction=float(np.abs(delta).max()),
            robot_domain=observe_native_arms(actions,self.joint_limits),
            actual_timing='Current-query event opportunity; one ARM chunk, base-only cooldown, rearm on new evidence')
        return actions,delta,base
