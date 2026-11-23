"""Frozen candidate, causal onset, and finite local ARM prescriptions."""
import numpy as np

from .causal_arm_gate import OneShotClosingGate
from .eval_diagnostics import array_hash
from .output_support_adapter import validate_native_arms
from .output_support_policy import OutputSupportPolicy
from .phase_target_policy import capture_all
from .structured_action_adapter import ARM_INDICES

CONDITIONS = ('Z_zero', 'F_first_full', 'B_two_full', 'H_two_half')
PRESCRIPTIONS = {'Z_zero': (0, 0.), 'F_first_full': (1, 1.),
                 'B_two_full': (2, 1.), 'H_two_half': (2, .5)}


def compose(base, raw, condition, query, event_window):
    if condition not in PRESCRIPTIONS or type(query) is not int or query < 0:
        raise ValueError('Registered condition and nonnegative query required')
    for value in (base, raw):
        if not isinstance(value, np.ndarray) or value.shape != (32, 14) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Finite float32 native chunks required')
    if np.max(np.abs(raw[:, ARM_INDICES])) > .100001:
        raise ValueError('Original ARM support exceeded')
    if event_window is not None and (len(event_window) != 2 or any(type(v) is not int for v in event_window) or event_window[0] < 1 or event_window[1] != event_window[0] + 2):
        raise ValueError('Original latched two-query opportunity required')
    duration, strength = PRESCRIPTIONS[condition]
    window = None if event_window is None else [event_window[0], event_window[0] + duration]
    active = window is not None and window[0] <= query < window[1]
    delta = np.zeros_like(base)
    if active:
        delta[:, ARM_INDICES] = raw[:, ARM_INDICES] * np.float32(strength)
    actions = base.copy()
    changed = delta != 0
    actions[changed] = base[changed] + delta[changed]
    return actions, delta, dict(window=window, active=bool(active), strength=strength if active else 0.)


class ArmPrescriptionPolicy(OutputSupportPolicy):
    def __init__(self, base, training_run, condition):
        if condition not in CONDITIONS:
            raise ValueError('Unknown local ARM prescription')
        self.detector = OneShotClosingGate()
        super().__init__(base, training_run, 'A_bounded_clipped', force_zero=True)
        self.condition = condition

    def reset_history(self):
        super().reset_history()
        self.detector.reset()

    def predict(self, observation, instruction, noise, *, enabled):
        if enabled is not True:
            raise ValueError('All conditions use the same frozen candidate')
        query, previous = self.query_count, self.previous
        _, _, base = super().predict(observation, instruction, noise, enabled=True)
        decision = self.detector.step(query, self.last_numeric['command_state'], base)
        actions, delta, prescription = compose(base, self.last_numeric['raw_delta'], self.condition, query, decision['window'])
        domain = validate_native_arms(actions, self.joint_limits)
        self.last_numeric.update(actions=actions.copy(), applied_delta=delta.copy())
        self.last_diagnostics.update(gate_decision=decision, prescription=prescription,
            window_active=prescription['active'], applied_window=prescription['window'],
            arm_strength=prescription['strength'], gripper_strength=0.,
            action_chunk_sha256=array_hash(actions), applied_delta_sha256=array_hash(delta),
            max_abs_correction=float(np.abs(delta).max()), applied_max_abs_correction=float(np.abs(delta).max()),
            robot_domain=domain, actual_timing='Current state/base causal onset; preregistered local duration/strength')
        capture_all(self, previous, base)
        return actions, delta, base
