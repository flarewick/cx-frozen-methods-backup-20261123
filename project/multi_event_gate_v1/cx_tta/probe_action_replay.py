"""Gate same-state interventions on deterministic, action-only scene replay."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from .audit import digest
from .eval_diagnostics import state_snapshot
from .evaluate_b2 import (TASKS, configure_runtime, read_legacy_l0,
                          resolve_tokenizer, seed_all)
from .paths import ROOT, new_run, write_json


def _physical_arrays(snapshot):
    arrays = {
        "arm_qpos": np.asarray(snapshot["arm_qpos"], dtype=np.float64),
        "gripper_qpos": np.asarray(snapshot["gripper_qpos"], dtype=np.float64),
    }
    for group in ("object_poses", "articulation_qpos"):
        for name, value in snapshot["offline_only"][group].items():
            arrays[f"{group}/{name}"] = np.asarray(value, dtype=np.float64)
    return arrays


def compare_physical(reference, actual, tolerance=1e-5):
    expected, observed = _physical_arrays(reference), _physical_arrays(actual)
    if set(expected) != set(observed):
        return {"matched": False, "reason": "entity_set_changed",
                "missing": sorted(set(expected) - set(observed)),
                "extra": sorted(set(observed) - set(expected))}
    errors = {}
    for name, value in expected.items():
        other = observed[name]
        if value.shape != other.shape or not np.isfinite(value).all() or not np.isfinite(other).all():
            return {"matched": False, "reason": "invalid_shape_or_value", "field": name}
        errors[name] = float(np.max(np.abs(value - other)))
    return {"matched": max(errors.values(), default=0.0) <= tolerance,
            "max_abs_error": max(errors.values(), default=0.0),
            "errors": errors, "tolerance": tolerance}


def load_source(source, seed=None):
    source = source.resolve()
    if not any(source.is_relative_to(ROOT / "outputs" / name)
               for name in ("diagnostics", "evaluations")):
        raise ValueError("Source must be a completed project evaluation or diagnostic")
    manifest = json.loads((source / "manifest.json").read_text())
    result = json.loads((source / "result.json").read_text())
    episodes = json.loads((source / "episodes.json").read_text())
    args = manifest["args"]
    if (result["status"] != "complete" or args["condition"] != "L0"
            or args["rollout_protocol"] != "official" or args["inference_mode"] != "compiled"
            or args["task"] not in TASKS or not args["trace"] or args["repeats"] != 1):
        raise ValueError("Source must be a completed traced compiled/official L0 run")
    if seed is None:
        if len(episodes) != 1:
            raise ValueError("Multi-episode source requires an explicit --seed")
        episode = episodes[0]
    else:
        matched = [row for row in episodes if row["seed"] == seed]
        if len(matched) != 1:
            raise ValueError("Requested seed missing or duplicated in source")
        episode = matched[0]
    if not episode["eligible"]:
        raise ValueError("Requested source episode is ineligible")
    trace_path = source / "traces" / f"r0_seed{episode['seed']}.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if (len(rows) != episode["inference_chunks"]
            or sum(row["executed"] for row in rows) != episode["executed_actions"]
            or any(row["noise_source"] != "model_internal_torch" or row["observation_updates"] != 32
                   or len(row["base_executed"]) != row["executed"] for row in rows)):
        raise ValueError("Source trace violates official L0 action contract")
    return manifest, episode, rows


def source_scene_index(manifest, seed):
    legacy = manifest["args"].get("legacy_l0")
    if legacy:
        legacy_path = Path(legacy)
        if not legacy_path.is_absolute():
            legacy_path = ROOT / legacy_path
        source_seeds = list(read_legacy_l0(legacy_path, manifest["args"]["task"]))
    else:
        source_seeds = list(manifest["seeds"])
    return source_seeds.index(seed)


def source_runtime(source):
    """Reuse the source run's frozen runtime snapshot without a large copy."""
    runtime = source / "robotwin"
    hashes = json.loads((source / "runtime_hashes.json").read_text())
    for name, key in (("envs/_base_task.py", "base_task_sha256"),
                      ("task_config/demo_randomized.yml", "task_config_sha256")):
        if digest(runtime / name) != hashes[key]:
            raise ValueError(f"Source runtime changed: {name}")
    return runtime


def prime_warp_cache(source, run):
    """Use a private copy of the source run's small compiled Warp kernels."""
    cached = source / "cache/warp_cache_path"
    if not cached.is_dir():
        raise FileNotFoundError(f"Source Warp cache missing: {cached}")
    shutil.copytree(cached, run / "cache/warp_cache_path")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--through-chunk", type=int, default=None,
                        help="Stop after validating this zero-based chunk; default checks full episode")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if not args.gpu.isdigit() or not 0 < args.tolerance <= 1e-3:
        parser.error("Require GPU index and tolerance in (0, 1e-3]")
    source = args.source.resolve()
    manifest, episode, rows = load_source(source, args.seed)
    scene_index = source_scene_index(manifest, episode["seed"])
    if args.through_chunk is not None:
        if not 0 <= args.through_chunk < len(rows):
            parser.error("--through-chunk must select an existing chunk")
        selected = rows[:args.through_chunk + 1]
    else:
        selected = rows
    task_name = manifest["args"]["task"]
    run = new_run("replay", group=("intervention_gates", task_name))
    print(run, flush=True)
    write_json(run / "manifest.json", {
        "source": str(source), "task": task_name, "seed": episode["seed"],
        "source_manifest_sha256": digest(source / "manifest.json"),
        "source_trace_sha256": digest(source / "traces" / f"r0_seed{episode['seed']}.jsonl"),
        "collector_sha256": digest(Path(__file__)), "through_chunk": args.through_chunk,
        "tolerance": args.tolerance, "purpose": "physical prefix replay gate; no policy query"})
    task = None
    try:
        runtime = source_runtime(source)
        prime_warp_cache(source, run)
        configure_runtime(run, runtime, resolve_tokenizer())
        config = json.loads((source / "resolved_config.json").read_text())
        config.update(save_path=str(run / "expert_data"), cx_runtime=str(runtime),
                      save_data=False, eval_video_log=False, eval_video_save_dir=None)
        task = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
        seed_all(episode["seed"])
        task.setup_demo(now_ep_num=scene_index, seed=episode["seed"],
                        **dict(config, need_plan=True, is_test=True,
                               save_path=str(run / "expert_data" / f"seed_{episode['seed']}")))
        task.set_instruction(instruction=episode["instruction"])
        task.get_obs()
        checkpoints = []
        initial = compare_physical(rows[0]["before"], state_snapshot(task), args.tolerance)
        checkpoints.append({"chunk": 0, "position": "initial", **initial})
        if initial["matched"]:
            for index, row in enumerate(selected):
                before = compare_physical(row["before"], state_snapshot(task), args.tolerance)
                checkpoints.append({"chunk": index, "position": "before", **before})
                if not before["matched"]:
                    break
                for action in row["base_executed"]:
                    previous = task.take_action_cnt
                    task.take_action(np.asarray(action, dtype=np.float32))
                    task.get_obs()
                    if task.take_action_cnt != previous + 1:
                        raise RuntimeError("Replay action did not advance exactly once")
                after = compare_physical(row["after"], state_snapshot(task), args.tolerance)
                checkpoints.append({"chunk": index, "position": "after", **after})
                if not after["matched"]:
                    break
        passed = (len(checkpoints) == 1 + 2 * len(selected)
                  and all(x["matched"] for x in checkpoints)
                  and (len(selected) < len(rows) or bool(task.eval_success) == episode["success"]))
        report = {"status": "complete", "passed": passed,
                  "source_chunks": len(rows), "checked_chunks": len(selected),
                  "checkpoints": checkpoints, "replayed_actions": int(task.take_action_cnt),
                  "original_actions": sum(row["executed"] for row in selected),
                  "full_episode_actions": episode["executed_actions"],
                  "replay_success": bool(task.eval_success), "original_success": episode["success"]}
        write_json(run / "result.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "checkpoints"}), flush=True)
        if not passed:
            raise RuntimeError("Physical checkpoint mismatch; counterfactual replay is not valid")
    except BaseException:
        write_json(run / "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        if task is not None and hasattr(task, "scene"):
            task.close_env()


if __name__ == "__main__":
    main()
