"""Paired RoboTwin L0/B2 evaluation on an explicit, fixed set of scene seeds."""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import traceback
import time
import numpy as np
from .audit import digest
from .cache_expert import CHECKPOINT, RT, create_base_policy
from .capture import instruction_for
from .collect_smoke import ROBOTWIN, make_config, configure_compiler_temp, write_guard
from .deploy import PolicyWithAdapter, load_adapter
from .paths import ROOT, new_run, write_json, replace_json

BEST = ROOT / "outputs/adapter_train_20260926_033411_615884/best.pt"
EXPECTED_BEST_SHA256 = "ef69cfc69fa6aa1d920fcefd81832bd2b6ff366025d99bd1dd91d67f0040c8af"
EVAL_SEEDS = tuple(range(310000, 310032))
TASKS = ("beat_block_hammer", "place_shoe", "turn_switch", "place_empty_cup")
PROJECT_TOKENIZER = (ROOT / "outputs/base_cache_20260926_024808_872087/cache/openpi"
                     / "big_vision/paligemma_tokenizer.model")


def resolve_tokenizer(explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer specified by --tokenizer-path does not exist: {path}")
        return path
    for path in (PROJECT_TOKENIZER, Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Tokenizer missing. Expected shared project copy at {PROJECT_TOKENIZER}; "
                            "otherwise pass --tokenizer-path explicitly")


def read_legacy_l0(path, task):
    """Accept only the audited 100380-100399 baseline band, with complete row data."""
    path = path.resolve()
    legacy_root = (ROOT.parent / "direction_guide_TTA_v5/outputs").resolve()
    if not path.is_relative_to(legacy_root) or path.name != "episodes.jsonl":
        raise ValueError("Legacy L0 must be an audited old-project episodes.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) > 20:
        raise ValueError("Legacy L0 is empty or outside the approved 20-seed diagnostic")
    result = {}
    for row in rows:
        seed = row["seed"]
        if (row["cond"] != "L0" or row["task"] != task or
                not isinstance(seed, int) or not 100380 <= seed <= 100399 or
                not isinstance(row["success"], bool) or
                not isinstance(row.get("instruction"), str) or not row["instruction"] or
                row.get("n_applied", 0) != 0 or seed in result):
            raise ValueError(f"Invalid/duplicate historical L0 episode: {seed}")
        result[seed] = row
    if list(result) != sorted(result):
        raise ValueError("Historical L0 rows must be sorted by seed")
    return result


def prepare_runtime(run):
    """Copy unmodified RoboTwin runtime into the new run; assets remain read-only."""
    runtime = run / "robotwin"
    runtime.mkdir()
    for name in ("envs", "task_config", "description"):
        shutil.copytree(ROBOTWIN / name, runtime / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".cache"))
    (runtime / "assets").symlink_to(ROBOTWIN / "assets", target_is_directory=True)
    write_json(run / "runtime_hashes.json", {
        "base_task_sha256": digest(runtime / "envs/_base_task.py"),
        "task_config_sha256": digest(runtime / "task_config/demo_randomized.yml"),
    })
    return runtime


def compiler_cache_paths(run):
    """Keep mutable autotune files private to one evaluation process."""
    return {key: run / "cache" / key.lower()
            for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")}


def configure_runtime(run, runtime, tokenizer_path):
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    for key in ("XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
                "CUDA_CACHE_PATH", "NUMBA_CACHE_DIR", "WARP_CACHE_PATH"):
        folder = run / "cache" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    # PyTorch autotune writes .best_config with exclusive create; sharing this
    # directory across concurrent GPU jobs can raise FileExistsError.
    for key, folder in compiler_cache_paths(run).items():
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    configure_compiler_temp(run)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["OPENPI_DATA_HOME"] = str(run / "cache/openpi")
    target = run / "cache/openpi/big_vision/paligemma_tokenizer.model"
    target.parent.mkdir(parents=True)
    shutil.copyfile(tokenizer_path, target)
    sys.addaudithook(write_guard)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_inference(mode, init_seed=None):
    """Explicit diagnostic protocol; default preserves the original compiled path."""
    if mode not in ("compiled", "eager"):
        raise ValueError("Unknown inference mode")
    import torch
    if mode == "eager":
        torch._dynamo.config.disable = True
    if init_seed is not None:
        seed_all(init_seed)


def step_chunk(task, wrapper, observation, instruction, rng, enabled, step_size=32, trace=None):
    if trace is not None:
        from .eval_diagnostics import state_snapshot
        before_state = state_snapshot(task)
    noise = rng.standard_normal((32, 32)).astype(np.float32)
    started = time.perf_counter()
    actions, delta, base = wrapper.predict(observation, instruction, noise, enabled=enabled)
    inference_s = time.perf_counter() - started
    executed = 0
    max_delta = 0.0
    for action, correction in zip(actions[:step_size], delta[:step_size]):
        if task.eval_success or task.take_action_cnt >= task.step_lim:
            break
        before = task.take_action_cnt
        task.take_action(action)
        if task.take_action_cnt == before:
            break
        executed += 1
        max_delta = max(max_delta, float(np.max(np.abs(correction))))
    if trace is not None:
        trace.record(observation=observation, before=before_state, after=state_snapshot(task),
                     noise=noise, base=base, actions=actions, delta=delta,
                     executed=executed, inference_s=inference_s)
    return executed, max_delta


def step_chunk_official(task, wrapper, observation, instruction, enabled, step_size=32, trace=None):
    """Match the local Pi0.5 deployment callback: internal noise and post-action get_obs."""
    if trace is not None:
        from .eval_diagnostics import state_snapshot
        before_state = state_snapshot(task)
    started = time.perf_counter()
    actions, delta, base = wrapper.predict(observation, instruction, None, enabled=enabled)
    inference_s = time.perf_counter() - started
    executed = 0
    observation_updates = 0
    max_delta = 0.0
    current_observation = observation
    for action, correction in zip(actions[:step_size], delta[:step_size]):
        before = task.take_action_cnt
        task.take_action(action)
        if task.take_action_cnt != before:
            executed += 1
            max_delta = max(max_delta, float(np.max(np.abs(correction))))
        current_observation = task.get_obs()
        observation_updates += 1
    if trace is not None:
        trace.record(observation=observation, before=before_state, after=state_snapshot(task),
                     noise=None, base=base, actions=actions, delta=delta,
                     executed=executed, inference_s=inference_s,
                     observation_updates=observation_updates)
    return executed, max_delta, current_observation


def run_episode(task, config, runtime, seed, index, wrapper, enabled, instruction_type="unseen",
                instruction_override=None, trace=None, rollout_protocol="paired"):
    """Follow official expert-valid-scene screening, then reset the identical seed."""
    opts = dict(config, need_plan=True, save_data=False, is_test=True,
                save_path=str(Path(config["save_path"]) / f"seed_{seed}"))
    if instruction_override is None:
        setup_started = False
        try:
            seed_all(seed)
            setup_started = True
            task.setup_demo(now_ep_num=index, seed=seed, **opts)
            scene_info = task.play_once()
            valid = bool(task.plan_success and task.check_success())
        finally:
            if setup_started:
                failed = sys.exc_info()[0] is not None
                try:
                    task.close_env()
                except Exception:
                    if not failed:
                        raise
        if not valid:
            return {"seed": seed, "eligible": False, "reason": "expert_plan_failed"}
        instruction = instruction_for(runtime, config["task_name"], scene_info, seed,
                                      split=instruction_type)
    else:
        # Historical L0 already fixed eligibility and language for these seeds.
        instruction = instruction_override
    setup_started = False
    try:
        seed_all(seed)
        setup_started = True
        task.setup_demo(now_ep_num=index, seed=seed, **opts)
        task.set_instruction(instruction=instruction)
        rng = np.random.default_rng(seed)
        chunks = 0
        max_delta = 0.0
        if rollout_protocol not in ("paired", "official"):
            raise ValueError("Unsupported rollout protocol")
        observation = task.get_obs() if rollout_protocol == "official" else None
        while task.take_action_cnt < task.step_lim and not task.eval_success:
            if rollout_protocol == "official":
                executed, delta, observation = step_chunk_official(
                    task, wrapper, observation, instruction, enabled, trace=trace)
            else:
                observation = task.get_obs()
                executed, delta = step_chunk(task, wrapper, observation, instruction, rng, enabled, trace=trace)
            if executed == 0:
                raise RuntimeError("Policy made no progress")
            chunks += 1
            max_delta = max(max_delta, delta)
        return {"seed": seed, "eligible": True, "success": bool(task.eval_success),
                "instruction": instruction, "executed_actions": int(task.take_action_cnt),
                "inference_chunks": chunks, "max_abs_correction": max_delta}
    finally:
        if setup_started:
            failed = sys.exc_info()[0] is not None
            try:
                task.close_env(clear_cache=(index + 1) % config["clear_cache_freq"] == 0)
            except Exception:
                if not failed:
                    raise


def format_eval_summary(task, condition, summary):
    eligible = summary["eligible_episodes"]
    attempted = summary["attempted_seeds"]
    rate = f'{summary["success_rate"]:.1%}' if summary["success_rate"] is not None else "N/A"
    return (f"评估完成 | 任务: {task} | 条件: {condition} | "
            f"成功: {summary['successes']}/{eligible} | 成功率: {rate} | "
            f"总尝试: {attempted} | 跳过: {attempted - eligible}")


def select_diagnostic_seeds(legacy, text, repeats, trace):
    if not 1 <= repeats <= 3:
        raise ValueError("repeats must be 1..3")
    if (text is not None or repeats > 1) and (legacy is None or not trace):
        raise ValueError("Explicit seeds/repeats require --legacy-l0 and --trace")
    if text is None:
        return tuple(legacy) if legacy is not None else None
    seeds = tuple(int(s) for s in text.split(","))
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in legacy for s in seeds):
        raise ValueError("Diagnostic seeds must be unique members of the historical seed list")
    return seeds


def write_provenance(run, runtime, tokenizer):
    import platform
    import importlib.metadata
    from .audit import files
    roots = {"project": ROOT / "cx_tta", "runtime": runtime / "envs",
             "openpi": RT / "RoboTwin/policy/pi05/src/openpi", "integration": RT / "pi05_integration"}
    sources = {f"{name}/{p.relative_to(folder)}": digest(p)
               for name, folder in roots.items() for p in files(folder) if p.suffix == ".py"}
    versions = {}
    for package in ("torch", "numpy", "sapien", "transformers", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(run / "provenance.json", {
        "python": platform.python_version(), "packages": versions, "source_sha256": sources,
        "base_checkpoint_sha256": digest(CHECKPOINT / "model.safetensors"),
        "normalization_sha256": digest(CHECKPOINT / "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"),
        "tokenizer_sha256": digest(tokenizer), "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "randomness": "python/numpy/torch seeded per episode; policy noise source recorded in manifest rollout_protocol"})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--condition", choices=("L0", "B2"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    parser.add_argument("--task-config", default="demo_randomized", choices=("demo_randomized",))
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--inference-mode", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--rollout-protocol", choices=("paired", "official"), default="paired",
                        help="official: internal Torch noise and get_obs after every executed action")
    parser.add_argument("--policy-init-seed", type=int, default=None,
                        help="Optional fixed model initialization seed for controlled process comparisons")
    parser.add_argument("--seed-start", type=int, default=310000)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--legacy-l0", type=Path,
                        help="Replay exact seed/instruction rows from a completed historical L0 JSONL for either condition")
    parser.add_argument("--trace", action="store_true", help="Log compact passive state/action diagnostics without video")
    parser.add_argument("--seeds", help="Comma-separated subset of historical seeds; requires --trace")
    parser.add_argument("--repeats", type=int, default=1, help="1..3 same-process diagnostic repetitions; requires --trace")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.gpu.isdigit() or args.episodes < 1 or args.episodes > 100 or args.seed_start < 0:
        parser.error("Invalid GPU/seed count")
    if args.policy_init_seed is not None and not 0 <= args.policy_init_seed < 2**32:
        parser.error("Policy initialization seed must be in [0, 2**32)")
    if args.rollout_protocol == "official" and args.inference_mode != "compiled":
        parser.error("Official rollout requires the compiled Pi0.5 backend")
    legacy = None
    if args.legacy_l0 is not None:
        legacy = read_legacy_l0(args.legacy_l0, args.task)
        seeds = tuple(legacy)
    else:
        seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
        if any(200000 <= seed < 250000 for seed in seeds):
            parser.error("Do not evaluate on development or demonstration seed bands")
    try:
        selected = select_diagnostic_seeds(legacy, args.seeds, args.repeats, args.trace)
        if selected is not None:
            seeds = selected
    except ValueError as exc:
        parser.error(str(exc))
    if not args.checkpoint.is_file() or digest(args.checkpoint) != EXPECTED_BEST_SHA256:
        parser.error("Frozen shared B2 checkpoint missing or changed")
    try:
        args.tokenizer_path = resolve_tokenizer(args.tokenizer_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    task_source = ROBOTWIN / "envs" / f"{args.task}.py"
    if not task_source.is_file():
        parser.error(f"RoboTwin task source missing: {task_source}")
    if args.dry_run:
        print(json.dumps({"task": args.task, "condition": args.condition, "seeds": seeds,
                          "checkpoint_sha256": EXPECTED_BEST_SHA256,
                          "frozen_base_checkpoint": str(CHECKPOINT),
                          "tokenizer_path": str(args.tokenizer_path),
                          "gpu_started": False, "paired_seed_protocol": True,
                          "repeats": args.repeats, "trace": args.trace,
                          "inference_mode": args.inference_mode, "rollout_protocol": args.rollout_protocol,
                          "policy_init_seed": args.policy_init_seed,
                          "legacy_l0": str(args.legacy_l0) if legacy else None}, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    diagnostic = args.seeds is not None or args.repeats > 1
    run = new_run("repro" if diagnostic else "eval",
                  group=("diagnostics" if diagnostic else "evaluations", args.task, args.condition))
    print(run, flush=True)
    write_json(run / "manifest.json", {"args": {**vars(args), "checkpoint": str(args.checkpoint),
                                              "tokenizer_path": str(args.tokenizer_path),
                                              "legacy_l0": str(args.legacy_l0) if args.legacy_l0 else None},
                                       "seeds": seeds, "checkpoint_sha256": EXPECTED_BEST_SHA256,
                                       "scope": "historical scene replay diagnostic" if legacy else
                                                "paired closed-loop L0/B2 diagnostic",
                                       "legacy_l0_sha256": digest(args.legacy_l0) if legacy else None})
    try:
        runtime = prepare_runtime(run)
        configure_runtime(run, runtime, args.tokenizer_path)
        if args.trace:
            print("Hashing policy/runtime provenance...", flush=True)
            write_provenance(run, runtime, args.tokenizer_path)
        config = make_config(runtime, args, run)
        config.update(eval_mode=True, save_data=False, eval_video_log=False,
                      eval_video_save_dir=None, render_freq=0)
        write_json(run / "resolved_config.json", config)
        task_cls = getattr(importlib.import_module(f"envs.{args.task}"), args.task)
        task = task_cls()
        configure_inference(args.inference_mode, args.policy_init_seed)
        base = create_base_policy()
        if args.trace:
            import torch
            write_json(run / "hardware.json", {"gpu": torch.cuda.get_device_name(0),
                       "cuda": torch.version.cuda, "torch": torch.__version__,
                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "cudnn_deterministic": torch.backends.cudnn.deterministic,
                       "dynamo_disabled": torch._dynamo.config.disable,
                       "float32_matmul_precision": torch.get_float32_matmul_precision()})
        adapter = load_adapter(args.checkpoint) if args.condition == "B2" else None
        wrapper = PolicyWithAdapter(base, adapter)
        records = []
        for repeat, seed in ((r, s) for r in range(args.repeats) for s in seeds):
            print(f"Starting {args.condition}: seed={seed}, repetition={repeat + 1}/{args.repeats}", flush=True)
            index = list(legacy).index(seed) if legacy is not None else list(seeds).index(seed)
            trace = None
            if args.trace:
                from .eval_diagnostics import EpisodeTrace
                trace = EpisodeTrace(run / "traces" / f"r{repeat}_seed{seed}.jsonl")
            try:
                result = run_episode(task, config, runtime, seed, index, wrapper,
                                     enabled=args.condition == "B2",
                                     instruction_type=args.instruction_type,
                                     instruction_override=legacy[seed]["instruction"] if legacy else None,
                                     trace=trace, rollout_protocol=args.rollout_protocol)
            except Exception as exc:
                if type(exc).__name__ != "UnStableError":
                    raise
                result = {"seed": seed, "eligible": False, "reason": "unstable_scene"}
            if diagnostic:
                result["repeat"] = repeat
            records.append(result)
            replace_json(run / "episodes.json", records)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        eligible = [row for row in records if row["eligible"]]
        summary = {"status": "complete", "attempted_seeds": len(records),
                   "eligible_episodes": len(eligible),
                   "successes": sum(row["success"] for row in eligible),
                   "success_rate": sum(row["success"] for row in eligible) / len(eligible) if eligible else None}
        if diagnostic:
            summary.update(scope="repeatability diagnostic, not independent success trials",
                           unique_seeds=len(seeds), repeats=args.repeats)
        if args.repeats > 1:
            from .eval_diagnostics import summarize_repeats
            write_json(run / "repeatability.json", summarize_repeats(run, records, seeds, args.repeats))
        write_json(run / "result.json", summary)
        print(json.dumps(summary), flush=True)
        print(format_eval_summary(args.task, args.condition, summary), flush=True)
    except BaseException:
        write_json(run / "failure.json", {"status": "failed_or_interrupted",
                                          "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
