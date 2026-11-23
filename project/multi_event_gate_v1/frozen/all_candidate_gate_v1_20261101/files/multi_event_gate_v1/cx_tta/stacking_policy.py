"""New expert-initialized candidate, existing causal events, no learned stacking gate yet."""
from pathlib import Path
import json
import numpy as np
import torch
from .stacking_adapter import StackingAdapter,read_checkpoint
from .official_onset_domain import OfficialOutputSupportPolicy,observe_native_arms
from .cache_joint_prefix import encode
from .robot_kinematics import RobotFK
from .train_output_support import robot_limits
from .multi_event import EventConfig,EventDetector,LocalController,apply_arm_candidate
from .empirical_onset_gate import candidate_packet
from .phase_target_policy import capture_all
from .recovery_contract import rng_capture,rng_fingerprint
from .eval_diagnostics import array_hash
from .audit import digest


class StackingPolicy(OfficialOutputSupportPolicy):
    def __init__(self,base,training_run,condition,config=EventConfig()):
        if condition not in ('L0','Z_zero','M_first_full','M_event_full'):raise ValueError('No trained stacking selector available')
        checkpoint=Path(training_run)/'adapter_seed0.pt';payload=read_checkpoint(checkpoint)
        self.metadata=payload['metadata'];self.condition=condition;self.base_policy=base
        self.model=StackingAdapter(32,self.metadata['geometry_mean']);self.model.load_state_dict(payload['model'])
        self.model.eval().requires_grad_(False)
        self.projection=dict(np.load(self.metadata['projection_path'],allow_pickle=False))
        robot_config=json.loads(Path(self.metadata['robot_config_path']).read_text())
        self.robots=[RobotFK(robot_config,side) for side in ('left','right')]
        self.joint_limits=robot_limits(robot_config)
        self.encode_eager=torch.compiler.disable(encode);self.sampler=base._sample_actions
        self.query_recorder=None;self.output_mode='bounded';self.force_zero=True
        self.derived_checkpoint_sha256=digest(checkpoint)
        self.event_config=EventConfig(**dict(config.to_dict(),max_interventions=1)) if condition=='M_first_full' else config
        self.detector=EventDetector(self.event_config);self.controller=LocalController(self.event_config)
        self.reset_history()

    def reset_history(self):
        super().reset_history();self.detector.reset();self.controller.reset();self.last_event_packet=None

    def predict(self,observation,instruction,noise,*,enabled):
        if enabled!=(self.condition!='L0'):raise ValueError('Condition/enable mismatch')
        q,previous=self.query_count,self.previous
        _,_,base=super().predict(observation,instruction,noise,enabled=enabled)
        capture_all(self,previous,base)
        before=rng_fingerprint(rng_capture());raw=self.last_numeric['raw_delta']
        packet=candidate_packet(self.last_legal_inputs,raw)
        event,evidence=self.detector.step(q,packet)
        def admit(_packet,_context):return dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
        decision=self.controller.step(q,event,packet,admit,enabled=self.condition in ('M_first_full','M_event_full'))
        actions,delta=apply_arm_candidate(base,raw,decision['active'])
        if before!=rng_fingerprint(rng_capture()):raise ValueError('Event logic changed sampler RNG')
        contexts=[r['context'] for r in decision['records'] if 'context' in r]
        self.last_event_packet=dict(packet,event_context=np.asarray(contexts,np.float32)) if contexts else None
        self.last_numeric.update(actions=actions.copy(),applied_delta=delta.copy())
        self.last_diagnostics.update(multi_event=dict(event=event,evidence=evidence,**decision),
            prescription=dict(active=decision['active'],strength=float(decision['active']),window=[q,q+1]),
            candidate_stage='expert_initialization_current_only',scoring_scope='always_admit_ablation_no_trained_gate',
            history_role='Actual causal history for event detector only; adapter masks all past fields in both train and deployment',
            arm_strength=float(decision['active']),gripper_strength=0.,
            action_chunk_sha256=array_hash(actions),applied_delta_sha256=array_hash(delta),
            max_abs_correction=float(np.abs(delta).max()),robot_domain=observe_native_arms(actions,self.joint_limits))
        return actions,delta,base
