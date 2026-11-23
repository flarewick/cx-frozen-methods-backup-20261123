"""G1 measurement contract, independent of tasks and simulator globals.

These measurements are NOT grasp labels or causal intervention effects.
The caller must estimate sensor positions or explicitly mark privileged data.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def frozen_array(value, shape, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Frame:
    episode_id: str
    step: int
    coordinate_frame: str
    source: str
    object_ids: tuple[str, ...]
    arm_ids: tuple[str, ...]
    object_xyz_m: np.ndarray
    tcp_xyz_m: np.ndarray
    gripper_open_fraction: np.ndarray
    object_confidence: np.ndarray
    arm_confidence: np.ndarray

    def __post_init__(self):
        if not self.episode_id or not self.coordinate_frame or type(self.step) is not int or self.step < 0:
            raise ValueError("Require episode, coordinate frame and nonnegative integer step")
        if self.source not in {"sensor_estimate", "simulator_gt"}:
            raise ValueError("Explicit observation provenance required")
        for key in ("object_ids", "arm_ids"):
            ids = tuple(getattr(self, key))
            if not ids or len(set(ids)) != len(ids) or not all(isinstance(s, str) and s for s in ids):
                raise ValueError(f"Invalid or duplicate {key}")
            object.__setattr__(self, key, ids)
        o, a = len(self.object_ids), len(self.arm_ids)
        for key, shape in [("object_xyz_m", (o, 3)), ("tcp_xyz_m", (a, 3)),
                           ("gripper_open_fraction", (a,)), ("object_confidence", (o,)),
                           ("arm_confidence", (a,))]:
            object.__setattr__(self, key, frozen_array(getattr(self, key), shape, key))
        for key in ("object_confidence", "arm_confidence"):
            x = getattr(self, key)
            if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
                raise ValueError("Confidence must be finite in [0,1]")
        for positions, confidence in [(self.object_xyz_m, self.object_confidence),
                                       (self.tcp_xyz_m, self.arm_confidence)]:
            if not np.isfinite(positions[confidence > 0]).all():
                raise ValueError("Visible positions must be finite")
        observed_grip = self.gripper_open_fraction[self.arm_confidence > 0]
        if not np.isfinite(observed_grip).all() or np.any((observed_grip < 0) | (observed_grip > 1)):
            raise ValueError("Observed gripper opening must be in [0,1]")


def measured_motion(before: Frame, after: Frame, min_confidence=0.5, min_motion_m=1e-4):
    """Return object x arm measurements, keeping association and missing masks."""
    if before.episode_id != after.episode_id or after.step <= before.step:
        raise ValueError("Transition must advance time within one episode")
    if before.coordinate_frame != after.coordinate_frame or before.source != after.source:
        raise ValueError("Cannot mix coordinate frames or provenance")
    if set(before.object_ids) != set(after.object_ids) or set(before.arm_ids) != set(after.arm_ids):
        raise ValueError("Entity set changed; tracker must explicitly resolve identities")
    if not 0 < min_confidence <= 1 or not np.isfinite(min_motion_m) or min_motion_m <= 0:
        raise ValueError("Invalid measurement thresholds")
    oi = [after.object_ids.index(key) for key in before.object_ids]
    ai = [after.arm_ids.index(key) for key in before.arm_ids]
    obj0, obj1 = before.object_xyz_m, after.object_xyz_m[oi]
    tcp0, tcp1 = before.tcp_xyz_m, after.tcp_xyz_m[ai]
    obj_ok = np.minimum(before.object_confidence, after.object_confidence[oi]) >= min_confidence
    arm_ok = np.minimum(before.arm_confidence, after.arm_confidence[ai]) >= min_confidence
    valid = obj_ok[:, None] & arm_ok[None, :]
    delta_obj, delta_tcp = obj1 - obj0, tcp1 - tcp0
    relative_before = obj0[:, None, :] - tcp0[None, :, :]
    relative_after = obj1[:, None, :] - tcp1[None, :, :]
    tcp_norm2 = np.sum(delta_tcp ** 2, axis=-1)
    measurable = valid & (tcp_norm2[None, :] >= min_motion_m ** 2)
    projection = np.full(valid.shape, np.nan)
    dot = np.einsum("oi,ai->oa", delta_obj, delta_tcp)
    np.divide(dot, tcp_norm2[None, :], out=projection, where=measurable)
    drift = np.linalg.norm(relative_after - relative_before, axis=-1)
    drift[~valid] = np.nan
    distances = np.linalg.norm(relative_before, axis=-1)
    distances[~valid] = np.nan
    return {
        "object_ids": before.object_ids, "arm_ids": before.arm_ids,
        "valid": valid, "projection_valid": measurable,
        "distance_before_m": distances, "relative_drift_m": drift,
        "object_along_tcp_ratio": projection,
        "object_delta_m": np.where(obj_ok[:, None], delta_obj, np.nan),
        "tcp_delta_m": np.where(arm_ok[:, None], delta_tcp, np.nan),
        "gripper_before": np.where(arm_ok, before.gripper_open_fraction, np.nan),
        "gripper_after": np.where(arm_ok, after.gripper_open_fraction[ai], np.nan),
        "privileged": before.source == "simulator_gt",
    }


@dataclass(frozen=True)
class ActionSpec:
    """Explicit caller-provided convention; no guessed 14D ordering."""
    names: tuple[str, ...]
    units: tuple[str, ...]
    representation: str
    normalization_id: str

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "units", tuple(self.units))
        if not self.names or len(set(self.names)) != len(self.names) or len(self.units) != len(self.names):
            raise ValueError("Action channels and units must match and be unique")
        if not self.representation or not self.normalization_id or not all(self.units):
            raise ValueError("Action representation, units and normalization must be explicit")


def transition(before, after, base_action, final_action, spec: ActionSpec, *, allow_privileged=False):
    """Only executed action prefix is a label, including early termination."""
    motion = measured_motion(before, after)
    if motion["privileged"] and not allow_privileged:
        raise ValueError("Privileged states are forbidden in deployable/TTA records")
    base, final = np.asarray(base_action, dtype=np.float64), np.asarray(final_action, dtype=np.float64)
    if base.ndim != 2 or final.shape != base.shape or base.shape[1] != len(spec.names):
        raise ValueError("Action blocks must be [horizon, named channels] with identical shape")
    if not np.isfinite(base).all() or not np.isfinite(final).all():
        raise ValueError("Actions must be finite")
    executed = after.step - before.step
    if executed > base.shape[0]:
        raise ValueError("Observed interval exceeds the provided action block")
    return {"episode_id": before.episode_id, "step_before": before.step, "step_after": after.step,
            "executed_steps": executed, "requested_horizon": base.shape[0],
            "base_action": base[:executed].copy(), "executed_action": final[:executed].copy(),
            "actual_delta": (final[:executed] - base[:executed]).copy(),
            "spec": spec, "motion": motion}
