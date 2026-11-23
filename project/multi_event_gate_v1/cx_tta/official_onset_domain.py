"""Official simulator action boundary without changing frozen diagnostic actors.

RoboTwin take_action(qpos) passes finite joint goals to TOPP/native execution; it
does not abort the evaluation on a policy-side URDF test. Observe that domain
here, without clipping/resampling/altering actions. Residual caps remain frozen.

Three inherited predict functions use isolated globals with only the validator
binding replaced. Their bytecode/closures are identical to the audited actors;
no shared module, old class, checkpoint or in-flight process is monkey-patched.
"""
from types import FunctionType
import numpy as np

from .output_support_adapter import native_domain_stats
from .output_support_policy import OutputSupportPolicy
from .arm_prescription_policy import ArmPrescriptionPolicy
from .shared_onset_policy import SharedOnsetPolicy
from .structured_action_adapter import ARM_INDICES


def observe_native_arms(actions, limits):
    stats = native_domain_stats(actions, limits)  # Reject nonfinite targets as before.
    a, bounds = np.asarray(actions), np.asarray(limits)
    excess = np.maximum(np.maximum(bounds[:, 0] - a, a - bounds[:, 1]), 0)
    stats.update(arm_max_overshoot_rad=float(excess[..., list(ARM_INDICES)].max()),
                 policy_boundary='official_native_observe_only_no_projection')
    return stats


def _official_predict(original):
    namespace = dict(original.__globals__)
    namespace['validate_native_arms'] = observe_native_arms
    fn = FunctionType(original.__code__, namespace, original.__name__,
                      original.__defaults__, original.__closure__)
    fn.__kwdefaults__ = original.__kwdefaults__
    return fn


class OfficialOutputSupportPolicy(OutputSupportPolicy):
    predict = _official_predict(OutputSupportPolicy.predict)


class OfficialArmPrescriptionPolicy(ArmPrescriptionPolicy, OfficialOutputSupportPolicy):
    predict = _official_predict(ArmPrescriptionPolicy.predict)


class OfficialSharedOnsetPolicy(SharedOnsetPolicy, OfficialOutputSupportPolicy):
    predict = _official_predict(SharedOnsetPolicy.predict)


def policy_classes(mode):
    if mode == 'official':
        return OfficialArmPrescriptionPolicy, OfficialSharedOnsetPolicy, OfficialOutputSupportPolicy
    if mode == 'strict':
        return ArmPrescriptionPolicy, SharedOnsetPolicy, OutputSupportPolicy
    raise ValueError('Unknown native action domain mode')
