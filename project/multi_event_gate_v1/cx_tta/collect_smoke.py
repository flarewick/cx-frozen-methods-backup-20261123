"""Bounded expert collection with smoke and formal training-data profiles.

This is not L0/Ar/Ag evaluation and produces no policy success-rate claim.
No language-generation subprocess or old collection shell script is invoked.
"""
from __future__ import annotations
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import traceback
from .audit import digest, files
from .paths import ROOT, new_run, write_json, replace_json

ROBOTWIN = Path("/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin")


def validate_options(args):
    if not args.task.isidentifier() or not (ROBOTWIN / "envs" / f"{args.task}.py").is_file():
        raise ValueError("Unknown local RoboTwin task")
    if args.task_config not in {"demo_clean", "demo_randomized"}:
        raise ValueError("Unsupported task configuration")
    limit = 10 if getattr(args, "profile", "smoke") == "smoke" else 1000
    if not 1 <= args.episodes <= args.max_attempts <= limit or args.seed_start < 0:
        raise ValueError(f"Require 1 <= episodes <= max-attempts <= {limit}, seed >= 0")
    if not args.gpu.isdigit():
        raise ValueError("Use one nonnegative GPU index")


def prepare_runtime(run):
    """Copy source/resources into this run; large assets remain read-only references."""
    target = run / "robotwin"
    target.mkdir()
    hashes = {}
    for folder in ("envs", "task_config", "description"):
        source = ROBOTWIN / folder
        shutil.copytree(source, target / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".cache"))
        for path in files(target / folder):
            if path.suffix in {".py", ".yml", ".json", ".sh"}:
                hashes[str(path.relative_to(target))] = digest(path)
    (target / "assets").symlink_to(ROBOTWIN / "assets", target_is_directory=True)
    from .recording import instrument_runtime
    original = hashes["envs/_base_task.py"]
    step_calls = instrument_runtime(target)
    hashes["envs/_base_task.py"] = digest(target / "envs/_base_task.py")
    write_json(run / "instrumentation.json", {
        "original_base_sha256": original, "instrumented_base_sha256": hashes["envs/_base_task.py"],
        "step_calls_instrumented": step_calls,
        "recording_sha256": digest(ROOT / "cx_tta/recording.py"),
        "capture_sha256": digest(ROOT / "cx_tta/capture.py"),
    })
    write_json(run / "runtime_source_hashes.json", hashes)
    return target


def write_guard(event, args):
    """Prevent Python-side writes escaping the new project, including asset symlinks.

Not an OS sandbox for native code; native output paths are explicitly configured.
"""
    paths = []
    if event == "open":
        path, mode, flags = args
        writable = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writable:
            paths = [path]
    elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"}:
        paths = [args[0]]
    elif event in {"os.rename", "os.link"}:
        paths = list(args[:2])
    elif event == "os.symlink":
        paths = [args[1]]
    for path in paths:
        if isinstance(path, int):
            continue
        resolved = Path(os.fsdecode(path)).resolve()
        if resolved != Path("/dev/null") and not resolved.is_relative_to(ROOT):
            raise PermissionError(f"New-project write boundary: {resolved}")


def make_config(runtime, args, run):
    import yaml
    config = yaml.safe_load((runtime / "task_config" / f"{args.task_config}.yml").read_text())
    types = yaml.safe_load((runtime / "task_config/_embodiment_config.yml").read_text())
    embodiment = config["embodiment"]
    if len(embodiment) != 1:
        raise ValueError("Initial smoke supports one shared dual-arm embodiment only")
    robot_file = (runtime / types[embodiment[0]]["file_path"]).resolve()
    robot_config = yaml.safe_load((robot_file / "config.yml").read_text())
    for arm in ("left", "right"):
        config[f"{arm}_robot_file"] = str(robot_file)
        config[f"{arm}_embodiment_config"] = copy.deepcopy(robot_config)
    config.update(task_name=args.task, task_config=args.task_config, render_freq=0,
                  dual_arm_embodied=True, embodiment_name=embodiment[0],
                  save_path=str(run / "expert_data"), eval_video_log=False,
                  eval_video_save_dir=None, cx_runtime=str(runtime))
    # Preserve the original expert sampling frequency. It is not policy chunk time.
    return config


def inspect_hdf5(path):
    import h5py
    import numpy as np
    result = {}
    with h5py.File(path, "r") as handle:
        def visit(name, item):
            if isinstance(item, h5py.Dataset):
                result[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
        handle.visititems(visit)
        action = handle["joint_action/vector"]
        if len(action.shape) != 2 or action.shape[1] != 14 or action.shape[0] < 2:
            raise ValueError(f"Unexpected expert drive-target shape {action.shape}")
        if not np.isfinite(action[:]).all():
            raise ValueError("Nonfinite drive targets")
        rgb = [name for name in result if name.startswith("observation/") and name.endswith("/rgb")]
        if len(rgb) < 3 or any(result[name]["shape"][0] != action.shape[0] for name in rgb):
            raise ValueError("Need three synchronized camera streams")
    return result


def configure_compiler_temp(run):
    """NVRTC rejects long TMPDIR names; keep per-process scratch out of deep run paths."""
    parent = ROOT / "outputs" / "_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="c_", dir=parent)).resolve()
    if len(os.fsencode(folder)) >= 120:
        folder.rmdir()
        raise ValueError("Project path too long for NVRTC scratch; use a shorter project root")
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = str(folder)
    tempfile.tempdir = str(folder)
    write_json(run / "compiler_temp.json", {"path": str(folder), "run": str(run),
                                           "path_bytes": len(os.fsencode(folder))})
    return folder


def execute(args, run, runtime):
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    # All cache/temp output redirected before any GPU/simulator imports.
    for key, leaf in [("XDG_CACHE_HOME", "xdg"), ("HF_HOME", "hf"),
                      ("TORCH_HOME", "torch"), ("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                      ("CUDA_CACHE_PATH", "cuda"), ("TRITON_CACHE_DIR", "triton"),
                      ("NUMBA_CACHE_DIR", "numba")]:
        folder = run / "cache" / leaf
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    configure_compiler_temp(run)
    sys.addaudithook(write_guard)
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    config = make_config(runtime, args, run)
    write_json(run / "resolved_config.json", config)
    cls = getattr(importlib.import_module(f"envs.{args.task}"), args.task)
    task = cls()
    from .recording import install_recording
    install_recording(task, effect_schema=getattr(args, "effect_schema", False))

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return collect_attempts(task, config, args, run, seed_all)


def collect_attempts(task, config, args, run, seed_all):
    """Separated orchestration for CPU contract tests with a fake environment."""
    records, accepted = [], 0
    # Mirrors official plan-then-replay protocol, with finite attempts and no shell side effects.
    for seed in range(args.seed_start, args.seed_start + args.max_attempts):
        row = {"seed": seed, "episode_index": accepted, "plan_success": False,
               "replay_success": False, "format_valid": False}
        episode_path = run / "expert_data" / f"seed_{seed}"
        setup_started = False
        try:
            for replay in (False, True):
                seed_all(seed)
                opts = copy.deepcopy(config)
                opts.update(need_plan=not replay, save_data=replay, save_path=str(episode_path))
                setup_started = True
                task.setup_demo(now_ep_num=accepted, seed=seed, **opts)
                if replay:
                    trajectory = task.load_tran_data(accepted)
                    opts.update(left_joint_path=trajectory["left_joint_path"],
                                right_joint_path=trajectory["right_joint_path"])
                    task.set_path_lst(opts)
                info = task.play_once()
                success = bool(task.plan_success and task.check_success())
                row["replay_success" if replay else "plan_success"] = success
                if not replay and success:
                    task.save_traj_data(accepted)
                if replay:
                    row["scene_info"] = info
                    if getattr(task, "cx_recording_enabled", False):
                        from .capture import export_capture, validate_capture
                        h5 = export_capture(task, config["cx_runtime"], args.task, info, seed, accepted)
                        row["capture_validation"] = validate_capture(h5)
                    else:  # CPU fake environment only; real entry always installs recording.
                        task.merge_pkl_to_hdf5_video()
                        h5 = episode_path / "data" / f"episode{accepted}.hdf5"
                    row["hdf5"] = str(h5)
                    row["schema"] = inspect_hdf5(h5)
                    row["format_valid"] = True
                    row["split"] = "validation" if accepted % 6 == 5 else "train"
                    if success and getattr(args, "profile", "smoke") == "train" and not getattr(args, "keep_raw", False):
                        task.remove_data_cache()
                    # Keep raw cache during this format probe, including failed replays.
                task.close_env()
                setup_started = False
                if not success:
                    break
            if row["replay_success"] and row["format_valid"]:
                accepted += 1
        except Exception as exc:
            row["error"] = traceback.format_exc()
            # Infrastructure errors fail fast; do not burn a seed budget on broken imports/format.
            if type(exc).__name__ != "UnStableError":
                records.append(row)
                replace_json(run / "attempts.json", records)
                raise
            row["excluded_reason"] = "unstable_scene"
        finally:
            if setup_started:
                try:
                    task.close_env()
                except Exception:
                    pass
        records.append(row)
        replace_json(run / "attempts.json", records)
        if accepted >= args.episodes:
            break
    replace_json(run / "attempts.json", records)
    return {
        "status": "collection_complete" if accepted >= args.episodes and getattr(args, "profile", "smoke") == "train"
                  else ("format_probe_complete" if accepted >= args.episodes else "insufficient_expert_successes"),
        "accepted_episodes": accepted, "attempts": len(records), "training_ready": False,
        "action_semantics": "joint_action/vector = left drive targets(6), left gripper command, right drive targets(6), right gripper command",
        "limitations": [
            "Expert demonstrations only; no frozen-policy same-state actions or failure recovery supervision",
            "Expert sampling save_freq is not VLA chunk horizon; timing/alignment must be audited",
            ("Schema 3 additionally records offline articulation states and integrated effective transition times"
             if getattr(args, "effect_schema", False) else
             "Schema 2 records grounded language, measured arms/raw gripper joints, physics ticks and offline GT poses"),
            "Only successful expert trajectories support initialization; failure/correction coverage still needed",
        ],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--profile", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--effect-schema", action="store_true", help="Schema 3 offline effect fields, in separate collections_effect directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_options(args)
    if args.dry_run:
        config = make_config(ROBOTWIN, args, ROOT / "outputs" / "dry_run_not_created")
        robot = config["left_embodiment_config"]
        joints = robot["arm_joints_name"]
        if len(joints) != 2 or any(len(arm) != 6 for arm in joints):
            raise ValueError("Smoke HDF5 contract requires this 6+gripper,6+gripper embodiment")
        print(json.dumps({"mode": "dry_run", "arguments": vars(args),
                          "output_parent": str(ROOT / "outputs" / ("collections_effect" if args.effect_schema else "collections") / args.task / args.task_config)
                          if args.profile == "train" else str(ROOT / "outputs"),
                          "save_video": False, "gpu_started": False,
                          "embodiment": config["embodiment_name"], "arm_joint_names": joints,
                          "expert_save_freq": config["save_freq"],
                          "runtime": "copy envs/task_config/description into new run; assets read-only reference",
                          "not_evaluation": True}, ensure_ascii=False, indent=2))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run = (new_run("collect", group=("collections_effect", args.task, args.task_config)) if args.effect_schema else
           (new_run("expert_smoke") if args.profile == "smoke" else
            new_run("collect", group=("collections", args.task, args.task_config))))
    print(run, flush=True)
    write_json(run / "manifest.json", {"arguments": vars(args), "source_root": str(ROBOTWIN),
                                      "entry_sha256": digest(Path(__file__)),
                                      "purpose": ("expert effect-contract data; not counterfactual interventions or failure recovery"
                                                  if args.effect_schema else
                                                  "expert demonstrations for shared offline residual initialization"),
                                      "schema_version": 3 if args.effect_schema else 2, "training_ready": False,
                                      "save_video": False, "rgb_storage": "JPEG frames inside HDF5",
                                      "cleanup_success_raw_cache": args.profile == "train" and not args.keep_raw})
    try:
        runtime = prepare_runtime(run)
        result = execute(args, run, runtime)
        write_json(run / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return int(result["status"] not in {"format_probe_complete", "collection_complete"})
    except BaseException:
        write_json(run / "failure.json", {"status": "failed_or_interrupted", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
