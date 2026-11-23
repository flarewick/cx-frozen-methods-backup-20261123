"""One-query, native-action deployment wrapper for the shared B2 baseline."""
from pathlib import Path
import cv2
import numpy as np
import torch
from .dataset import context_features, CONTEXT_KIND
from .residual import SharedResidualAdapter

CAMERA_KEYS = ("head_camera", "right_camera", "left_camera")
POLICY_CAMERA_KEYS = ("cam_high", "cam_right_wrist", "cam_left_wrist")


def unpack_observation(observation):
    images = [np.asarray(observation["observation"][name]["rgb"], dtype=np.uint8)
              for name in CAMERA_KEYS]
    if any(im.ndim != 3 or im.shape[2] != 3 for im in images):
        raise ValueError("Expected three HWC RGB images")
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 14D native command state")
    return images, state


def training_image_roundtrip(image):
    """Match the collector's JPEG RGB representation for adapter context only."""
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decoding failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def load_adapter(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata["scope"] != "B2 shared residual BC, not direction/effect/TTA":
        raise ValueError("Unexpected model scope")
    model = SharedResidualAdapter(590, metadata["caps_native_units"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


class PolicyWithAdapter:
    def __init__(self, base_policy, adapter=None):
        self.base_policy = base_policy
        self.adapter = adapter

    def predict(self, observation, prompt, noise, *, enabled):
        images, state = unpack_observation(observation)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Grounded instruction required")
        policy_obs = {"state": state, "prompt": prompt,
                      "images": {name: image.transpose(2, 0, 1)
                                 for name, image in zip(POLICY_CAMERA_KEYS, images)}}
        # Exactly one frozen-policy query for both conditions; returned actions are native 14D.
        base = np.asarray(self.base_policy.infer(policy_obs, noise=noise)["actions"], dtype=np.float32)
        if base.shape != (32, 14) or not np.isfinite(base).all():
            raise ValueError("Unexpected frozen-policy action chunk")
        if not enabled:
            return base, np.zeros_like(base), base
        if self.adapter is None:
            raise ValueError("Adapter condition requires a checkpoint")
        context = context_features([training_image_roundtrip(im) for im in images], state)
        with torch.no_grad():
            output, delta = self.adapter(torch.from_numpy(context)[None],
                                         torch.from_numpy(base)[None])
        action = output[0].numpy()
        correction = delta[0].numpy()
        if not np.isfinite(action).all() or not np.isfinite(correction).all():
            raise ValueError("Nonfinite corrected action")
        return action, correction, base
