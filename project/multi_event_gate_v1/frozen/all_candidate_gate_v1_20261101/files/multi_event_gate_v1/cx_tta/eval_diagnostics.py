"""Passive, compact evaluation traces. Privileged fields never enter the policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .paths import output_path
from .recording import measured_state, object_poses


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def state_snapshot(task):
    """Read physical state without stepping, rendering, or querying task success."""
    arms, grippers = measured_state(task.robot)
    articulations = {}
    for name, value in vars(task).items():
        if name.startswith("_"):
            continue
        objects = list(enumerate(value)) if isinstance(value, (list, tuple)) else [(None, value)]
        for index, obj in objects:
            if hasattr(obj, "config") and hasattr(obj, "actor") and callable(getattr(obj, "get_qpos", None)):
                key = name if index is None else f"{name}_{index}"
                articulations[key] = np.asarray(obj.get_qpos()).tolist()
    return {"arm_qpos": arms.tolist(), "gripper_qpos": grippers.tolist(),
            "offline_only": {"object_poses": {k: v.tolist() for k, v in object_poses(task).items()},
                             "articulation_qpos": articulations}}


class EpisodeTrace:
    def __init__(self, path):
        self.path = output_path(Path(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.count = 0

    def record(self, *, observation, before, after, noise, base, actions, delta, executed, inference_s,
               observation_updates=0):
        images = {name: array_hash(observation["observation"][name]["rgb"])
                  for name in ("head_camera", "right_camera", "left_camera")}
        obs = {"images": images, "command_state": array_hash(observation["joint_action"]["vector"])}
        row = {"chunk": self.count, "observation_sha256": json_hash(obs),
               "observation_components": obs, "state_before_sha256": json_hash(before),
               "before": before, "after": after,
               "noise_source": "model_internal_torch" if noise is None else "explicit_numpy",
               "noise_sha256": None if noise is None else array_hash(noise),
               "observation_updates": observation_updates,
               "base_chunk_sha256": array_hash(base), "executed": executed,
               "base_executed": np.asarray(base[:executed]).tolist(),
               "actions_executed": np.asarray(actions[:executed]).tolist(),
               "delta_executed": np.asarray(delta[:executed]).tolist(),
               "inference_s": inference_s}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.count += 1


def summarize_repeats(run, records, seeds, repeats):
    """Compare same-seed repetitions; this measures repeatability, not success gain."""
    pairs = []
    for seed in seeds:
        ref_result = next(r for r in records if r["seed"] == seed and r["repeat"] == 0)
        reference = [json.loads(line) for line in (run / "traces" / f"r0_seed{seed}.jsonl").read_text().splitlines()]
        for repeat in range(1, repeats):
            result = next(r for r in records if r["seed"] == seed and r["repeat"] == repeat)
            rows = [json.loads(line) for line in (run / "traces" / f"r{repeat}_seed{seed}.jsonl").read_text().splitlines()]
            first = lambda key: next((i for i, (a, b) in enumerate(zip(reference, rows)) if a[key] != b[key]), None)
            pairs.append({"seed": seed, "repeat": repeat,
                          "success_equal": ref_result.get("success") == result.get("success"),
                          "eligibility_equal": ref_result["eligible"] == result["eligible"],
                          "chunks": [len(reference), len(rows)],
                          "initial_state_equal": bool(reference and rows and reference[0]["state_before_sha256"] == rows[0]["state_before_sha256"]),
                          "initial_observation_equal": bool(reference and rows and reference[0]["observation_sha256"] == rows[0]["observation_sha256"]),
                          "first_observation_divergence_chunk": first("observation_sha256"),
                          "first_base_action_divergence_chunk": first("base_chunk_sha256"),
                          "first_noise_divergence_chunk": first("noise_sha256"),
                          "trajectory_equal": bool(reference and rows and len(reference) == len(rows)
                              and all(a["before"] == b["before"] and a["after"] == b["after"]
                                      and a["actions_executed"] == b["actions_executed"]
                                      and a["observation_sha256"] == b["observation_sha256"]
                                      for a, b in zip(reference, rows)))})
    return {"scope": "within-process repeatability; not independent success trials", "pairs": pairs}
