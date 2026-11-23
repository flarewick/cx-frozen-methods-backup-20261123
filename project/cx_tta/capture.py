"""Streaming export of locally generated trusted frame pickles; no ffmpeg dependency."""
import json
from pathlib import Path
import pickle
import random
import re
import cv2
import h5py
import numpy as np
from .paths import output_path

CAMERAS = ("head_camera", "right_camera", "left_camera")


def flatten(record, prefix=""):
    for name, value in record.items():
        key = f"{prefix}/{name}".strip("/")
        if isinstance(value, dict):
            yield from flatten(value, key)
        elif isinstance(value, (list, tuple, np.ndarray, float, int, np.number)):
            value = np.asarray(value)
            if value.dtype.kind not in "biuf" or value.size == 0:
                continue
            yield key, value


def instruction_for(runtime, task_name, scene_info, seed, *, split="seen"):
    """Ground a deterministic local language template in the planned scene."""
    if split not in {"seen", "unseen"}:
        raise ValueError("Instruction split must be seen or unseen")
    rng = random.Random(seed)
    params = {k.strip("{}"): str(v) for k, v in scene_info.get("info", {}).items()}
    folder = Path(runtime) / "description"
    template_data = json.loads((folder / "task_instruction" / f"{task_name}.json").read_text())
    candidates = [s for s in template_data[split]
                  if set(re.findall(r"\{([^}]+)\}", s)).issubset(params)]
    if not candidates:
        raise ValueError(f"No fully grounded instruction template for {task_name}")
    template = rng.choice(candidates)
    def fill(match):
        key = match.group(1)
        value = params[key]
        path = (folder / "objects_description" / (value + ".json")).resolve()
        if not path.is_relative_to((folder / "objects_description").resolve()):
            raise ValueError("Object description path escape")
        if path.is_file():
            choices = json.loads(path.read_text())[split]
            return "the " + rng.choice(choices)
        if "/" in value or "\\" in value:
            raise ValueError(f"Missing object description: {path}")
        return f"the {value} arm" if len(key) == 1 and key.islower() else value
    return re.sub(r"\{([^}]+)\}", fill, template)


def export_capture(task, runtime, task_name, scene_info, seed, episode_index):
    base = output_path(Path(task.save_dir))
    folder = output_path(Path(task.folder_path["cache"]))
    frames = sorted(folder.glob("*.pkl"), key=lambda p: int(p.stem))
    if len(frames) < 2 or [int(p.stem) for p in frames] != list(range(len(frames))):
        raise ValueError("Incomplete frame sequence")
    path = base / "data" / f"episode{episode_index}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = instruction_for(runtime, task_name, scene_info, seed)
    datasets, shapes = {}, {}
    with h5py.File(path, "x") as h:
        h.attrs.update(schema_version=3 if getattr(task, "cx_effect_enabled", False) else 2, task=task_name, seed=seed,
                       instruction=prompt, image_encoding="jpeg_rgb",
                       action_semantics="native_absolute_drive_targets_and_gripper_commands",
                       label_alignment="observation[i] -> expert_command[i+1]",
                       training_ready=False)
        for i, frame_path in enumerate(frames):
            # Only our local collector creates these pickles; never load arbitrary external pickles.
            with frame_path.open("rb") as handle:
                frame = pickle.load(handle)
            values = dict(flatten(frame))
            if i and set(values) != set(datasets):
                raise ValueError("Frame schema/entity set changed within episode")
            for name, value in values.items():
                if i == 0:
                    shapes[name] = value.shape
                    if name.endswith("/rgb"):
                        datasets[name] = h.create_dataset(name, (len(frames),),
                                                          dtype=h5py.vlen_dtype(np.dtype("uint8")))
                    else:
                        datasets[name] = h.create_dataset(name, (len(frames),) + value.shape,
                                                          dtype=value.dtype, compression="gzip")
                if value.shape != shapes[name] or not np.isfinite(value).all():
                    raise ValueError(f"Invalid frame {i} field {name}")
                if name.endswith("/rgb"):
                    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
                    if not ok:
                        raise ValueError(f"Failed JPEG encoding {name}")
                    datasets[name][i] = encoded.reshape(-1)
                else:
                    datasets[name][i] = value
        dense = task._cx_dense
        if not dense:
            raise ValueError("No actual physics command trace")
        h.create_dataset("cx_dense/tick_before", data=np.asarray([v[0] for v in dense], dtype=np.int64))
        h.create_dataset("cx_dense/dt_s", data=np.asarray([v[1] for v in dense]))
        h.create_dataset("cx_dense/command", data=np.stack([v[2] for v in dense]), compression="gzip")
        if getattr(task, "cx_effect_enabled", False):
            from .effect_contract import add_timeline
            add_timeline(h)
    return path


def validate_capture(path, decode_all=True):
    with h5py.File(path, "r") as h:
        if int(h.attrs.get("schema_version", 0)) not in (2, 3) or not h.attrs.get("instruction"):
            raise ValueError("Capture schema/instruction missing")
        actions = h["joint_action/vector"][:]
        ticks = h["cx_measurement/physics_tick"][:]
        arms = h["cx_measurement/arm_qpos"][:]
        if actions.ndim != 2 or actions.shape[1] != 14 or arms.shape != (len(actions), 2, 6):
            raise ValueError("Invalid native action or measured qpos dimensions")
        if len(actions) < 2 or len(ticks) != len(actions) or np.any(np.diff(ticks) < 0):
            raise ValueError("Nonmonotonic or missing time index")
        if not np.isfinite(actions).all() or not np.isfinite(arms).all():
            raise ValueError("Nonfinite actions/state")
        dense_ticks = h["cx_dense/tick_before"][:]
        dense_commands = h["cx_dense/command"][:]
        if len(dense_ticks) < 1 or dense_commands.shape != (len(dense_ticks), 14):
            raise ValueError("Dense command stream incomplete")
        if np.any(np.diff(dense_ticks) != 1) or dense_ticks[0] > ticks[0] or dense_ticks[-1] + 1 < ticks[-1]:
            raise ValueError("Dense trace does not cover observations")
        decoded = 0
        for camera in CAMERAS:
            ds = h[f"observation/{camera}/rgb"]
            if len(ds) != len(actions):
                raise ValueError("Unsynchronized RGB")
            for i in range(len(ds)) if decode_all else (0, len(ds) - 1):
                image = cv2.imdecode(np.asarray(ds[i], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3:
                    raise ValueError("Invalid JPEG")
                decoded += 1
        effect = {}
        if int(h.attrs["schema_version"]) == 3:
            from .effect_contract import validate_effect
            effect = {"effect_contract": validate_effect(h)}
        return {"frames": len(actions), "decoded_images": decoded, "dense_steps": len(dense_ticks), **effect,
                "objects_gt": list(h["cx_privileged"].keys()),
                "instruction": str(h.attrs["instruction"]), "schema_version": int(h.attrs["schema_version"]),
                "duplicate_frame_ticks": int(np.sum(np.diff(ticks) == 0)),
                "expert_data_valid": True, "residual_training_ready": False}
