"""Fixed causal feature groups for later shared benefit/risk readout comparisons.

Only the current observation, actual previous observation, current base plan,
and exact frozen candidate are used. FK is command FK, not a contact sensor.
This module fits nothing, chooses no thresholds, and reads no offline labels.
"""
import numpy as np
from .onset_observables import validate_packet
from .structured_action_adapter import ARM_INDICES


def feature_groups(packet):
    p=validate_packet(packet)
    history=p['has_history'][0,0]
    # Preserve camera/patch identity; pooling would discard contact-local cues.
    observation=np.concatenate([p[k].reshape(-1) for k in ('visual','text','state','ee_pose')])
    motion=np.concatenate([history*(p[k]-p['past_'+k]).reshape(-1)
        for k in ('visual','state','ee_pose')]+[np.asarray([history],np.float32)])
    state=p['state'][0].copy();state[list(ARM_INDICES)]*=np.float32(np.pi)
    # Native-radian planned changes from current commanded state, not actual motion.
    base=p['base'][0];delta=p['candidate_delta'][0]
    planned=base-state[None]
    knots=[0,8,16,24,31]
    base_plan=planned[knots].reshape(-1)
    candidate=delta[knots][:,ARM_INDICES].reshape(-1)
    arm_plan=planned[:,ARM_INDICES];arm_delta=delta[:,ARM_INDICES]
    denom=np.linalg.norm(arm_plan)*np.linalg.norm(arm_delta)
    cosine=float((arm_plan*arm_delta).sum()/denom) if denom>1e-12 else 0.
    summaries=np.asarray([np.sqrt(np.mean(arm_delta**2)),np.max(np.abs(arm_delta)),
        np.linalg.norm(arm_plan),cosine],np.float32)
    return {k:np.asarray(v,np.float32) for k,v in dict(observation=observation,
        observed_change=motion,base_plan=base_plan,candidate=candidate,candidate_interaction=summaries).items()}
