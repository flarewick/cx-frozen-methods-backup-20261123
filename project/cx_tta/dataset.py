"""Native expert labels following local pi05/scripts/process_data.py: action[i]=command[i+1]."""
from pathlib import Path
import cv2
import h5py
import numpy as np
from .capture import CAMERAS

CONTEXT_KIND = "native_rgb_grid8_command_state14_v1"


def read_observation(handle, index):
    if handle.attrs.get("image_encoding") != "jpeg_rgb":
        raise ValueError("Use schema2 JPEG RGB contract, not legacy ambiguous color encoding")
    images = []
    for camera in CAMERAS:
        raw = np.asarray(handle[f"observation/{camera}/rgb"][index], dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid image")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    state = np.asarray(handle["joint_action/vector"][index], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Bad native drive-target state")
    return images, state


def context_features(images, state):
    """B2 baseline only: cheap nonprivileged visual grid plus current command state.

Task semantics also enter through the instruction-conditioned frozen base action.
No task ID, future image, GT geometry, or success signal enters this vector.
"""
    state = np.asarray(state, np.float32).copy()
    state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] /= np.pi
    pooled = [cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 127.5 - 1
              for image in images]
    return np.concatenate([state, *pooled]).astype(np.float32)


def expert_chunk(commands, index, horizon=32):
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1] != 14 or not 0 <= index < len(commands) - 1:
        raise ValueError("Need current observation and at least one future expert command")
    count = min(horizon, len(commands) - index - 1)
    target = np.zeros((horizon, 14), dtype=np.float32)
    target[:count] = commands[index + 1:index + 1 + count]
    mask = np.arange(horizon) < count
    return target, mask


def load_cache(path):
    with np.load(path, allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    n = len(data["context"])
    if str(data["context_kind"]) != CONTEXT_KIND:
        raise ValueError("Unsupported or privileged feature contract")
    if data["context"].shape != (n, 590) or data["base_action"].shape != (n, 32, 14):
        raise ValueError("Invalid cached model input shape")
    if data["target"].shape != (n, 32, 14) or data["valid"].shape != (n, 32):
        raise ValueError("Invalid target/mask shape")
    if data["valid"].dtype != np.bool_ or not data["valid"].any(axis=1).all():
        raise ValueError("Invalid action masks")
    for key in ("context", "base_action", "target"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite cache {key}")
    return data
