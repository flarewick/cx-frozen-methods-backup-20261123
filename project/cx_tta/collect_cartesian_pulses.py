"""Schema 4: thirteen Cartesian pulse branches; offline physical labels.

Versioned from frozen collect_interventions.py; the original collector is unchanged.

Bounded robot FK proposes directions; simulator truth is used only as an
offline label. Each branch resets and replays the exact L0 action prefix.
"""
import argparse
import importlib
import json
from pathlib import Path
import traceback

import numpy as np

from .cartesian_pulse import BRANCHES, candidates, pulse_windows
from .robot_kinematics import RobotFK
from .short_window_contract import entity_ids, make_windows
from .pulse_contacts import capture_frame
from .audit import digest
from .deploy import unpack_observation
from .eval_diagnostics import array_hash, state_snapshot
from .effect_contract import articulation_state
from .intervention_contract import install_action_clock, validate_transition
from .evaluate_b2 import (configure_runtime,
                          resolve_tokenizer, seed_all)
from .paths import new_run, output_path, write_json
from .probe_action_replay import (compare_physical, load_source, prime_warp_cache,
                                  source_runtime, source_scene_index)


def observation_signature(observation):
    images, _ = unpack_observation(observation)
    # EpisodeTrace hashes the raw command array, including its original dtype.
    raw_state = np.asarray(observation["joint_action"]["vector"])
    return {"images": {name: array_hash(image) for name, image in zip(
                ("head_camera", "right_camera", "left_camera"), images)},
            "command_state": array_hash(raw_state)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Completed compiled/official L0 one-episode trace")
    parser.add_argument("--seed", type=int, default=None,
                        help="Required when the source run contains multiple episodes")
    parser.add_argument("--chunk", type=int, required=True,
                        help="Zero-based full 32-action chunk to intervene on")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="New directory under project outputs; must not already exist")
    parser.add_argument("--pulse-start", type=int, choices=(0,8,16,24), default=0)
    args = parser.parse_args()
    args.record_stride = 4
    args.proposal_set = "cartesian_pulse_v1"
    if not args.gpu.isdigit() or not 0 < args.tolerance <= 1e-3:
        parser.error("Invalid GPU or tolerance")
    source = args.source.resolve()
    manifest, episode, rows = load_source(source, args.seed)
    scene_index = source_scene_index(manifest, episode["seed"])
    if not 0 <= args.chunk < len(rows) or rows[args.chunk]["executed"] != 32:
        parser.error("Select a full 32-action L0 chunk; partial terminal chunks are excluded")
    task_name = manifest["args"]["task"]
    extended = True
    branch_names = BRANCHES
    if args.run_dir is None:
        run = new_run("windows", group=("cartesian_pulses", task_name))
    else:
        run = output_path(args.run_dir)
        run.mkdir(parents=True, exist_ok=False)
    print(run, flush=True)
    write_json(run / "manifest.json", {"source": str(source), "source_seed": episode["seed"],
                                      "task": task_name, "chunk": args.chunk,
                                      "instruction": episode["instruction"],
                                      "scene_group": f"{task_name}:{episode['seed']}",
                                      "source_manifest_sha256": digest(source / "manifest.json"),
                                      "source_trace_sha256": digest(source / "traces" / f"r0_seed{episode['seed']}.jsonl"),
                                      "collector_sha256": digest(Path(__file__)),
                                      "contact_code_sha256": digest(Path(__file__).with_name("pulse_contacts.py")),
                                      "source_config_sha256": digest(source / "resolved_config.json"),
                                      "proposal_code_sha256": digest(Path(__file__).with_name("cartesian_pulse.py")),
                                      "fk_code_sha256": digest(Path(__file__).with_name("robot_kinematics.py")),
                                      "solver_code_sha256": digest(Path(__file__).with_name("audit_cartesian_proposals.py")),
                                      "tolerance": args.tolerance,
                                      "schema_version": 4,
                  "record_stride": args.record_stride,
                                      "proposal_set": args.proposal_set,
                                      "pulse_start": args.pulse_start,
                                      "contract_sha256": digest(Path(__file__).with_name("intervention_contract.py")),
                                      "window_contract_sha256": digest(Path(__file__).with_name("short_window_contract.py")),
                                      "branches": list(branch_names),
                                      "scope": "simulator same-prefix intervention, not TTA or success-rate evaluation"})
    try:
        runtime = source_runtime(source)
        prime_warp_cache(source, run)
        configure_runtime(run, runtime, resolve_tokenizer())
        config = json.loads((source / "resolved_config.json").read_text())
        config.update(save_path=str(run / "expert_data"), cx_runtime=str(runtime),
                      save_data=False, eval_video_log=False, eval_video_save_dir=None)
        task_cls = getattr(importlib.import_module(f"envs.{task_name}"), task_name)
        if extended:
            write_json(run / "clock_instrumentation.json", install_action_clock(task_cls))
        robots = [RobotFK(config,side) for side in ("left","right")]
        base = np.asarray(rows[args.chunk]["base_executed"], dtype=np.float32)
        # The branch runner returns the checkpoint observation through this
        # observer only once; no base-policy query is needed for frozen actions.
        proposal = {}
        results = {}
        for name in branch_names:
            task = task_cls()
            try:
                seed_all(episode["seed"])
                task.setup_demo(now_ep_num=scene_index, seed=episode["seed"],
                                **dict(config, need_plan=True, is_test=True,
                                       save_path=str(Path(config["save_path"]) / f"seed_{episode['seed']}")))
                task.set_instruction(instruction=episode["instruction"])
                obs = task.get_obs()
                mapping, scene_ids = entity_ids(task)
                frames = []
                prefix = [a for r in rows[:args.chunk] for a in r["base_executed"]]
                prefix_count = 0
                for row in rows[:args.chunk]:
                    for action in row["base_executed"]:
                        previous = task.take_action_cnt
                        task.take_action(np.asarray(action, dtype=np.float32))
                        obs = task.get_obs()
                        if task.take_action_cnt != previous + 1:
                            raise RuntimeError("Prefix action did not advance exactly once")
                        prefix_count += 1
                        if prefix_count == len(prefix) - args.record_stride:
                            frames.append(capture_frame(task, obs, run, name, -args.record_stride, mapping, scene_ids))
                before = state_snapshot(task)
                match = compare_physical(rows[args.chunk]["before"], before, args.tolerance)
                if not match["matched"]:
                    raise RuntimeError(f"Branch start state mismatch: {match}")
                signature = observation_signature(obs)
                if name == "base":
                    source_observation_match = signature == rows[args.chunk]["observation_components"]
                    write_json(run / "observation_comparison.json", {
                        "source": rows[args.chunk]["observation_components"],
                        "replayed": signature, "exact_match": source_observation_match,
                        "interpretation": "Intervention branches require a common replayed observation; source image identity is separately reported"})
                    images, command = unpack_observation(obs)
                    proposal, specification, cartesian = candidates(base, robots, args.pulse_start)
                    write_json(run / "proposal_spec.json", specification)
                    np.savez_compressed(run / "checkpoint.npz", command_state=command, base_actions=base,
                                        head_rgb=images[0], right_rgb=images[1], left_rgb=images[2])
                    first_signature = signature
                elif signature != first_signature:
                    raise RuntimeError("Branch initial observation differs; candidate context is not shared")
                frames.append(capture_frame(task, obs, run, name, 0, mapping, scene_ids))
                masks = {k: v["semantic_mask_sha256"] for k, v in frames[-1]["visibility_offline_only"].items()}
                if name == "base":
                    first_masks = masks
                elif masks != first_masks:
                    raise RuntimeError("Initial semantic segmentation differs across branches")
                if name != "base":
                    common_base = {f["offset"]: f for f in results["base"]["frames"]}
                executed = 0
                executed_mask, durations, physics_steps = [], [], []
                if extended:
                    joint_limits = {k: v['qlimits'].tolist() for k, v in articulation_state(task).items()}
                for action_index, action in enumerate(proposal[name], start=1):
                    previous = task.take_action_cnt
                    time_before = getattr(task, '_cx_intervention_time', 0.0)
                    ticks_before = getattr(task, '_cx_intervention_ticks', 0)
                    task.take_action(action)
                    obs = task.get_obs()
                    advanced = task.take_action_cnt - previous
                    if advanced not in (0, 1):
                        raise RuntimeError("Branch action advanced by an unexpected count")
                    executed_mask.append(bool(advanced))
                    durations.append(getattr(task, '_cx_intervention_time', 0.0) - time_before)
                    physics_steps.append(getattr(task, '_cx_intervention_ticks', 0) - ticks_before)
                    if advanced:
                        executed += 1
                    if action_index % args.record_stride == 0 or (advanced and task.eval_success):
                        frames.append(capture_frame(task, obs, run, name, action_index, mapping, scene_ids))
                if name != "base":
                    for frame in frames:
                        if 0 <= frame["offset"] <= args.pulse_start:
                            other = common_base[frame["offset"]]
                            if frame["observation_signature"] != other["observation_signature"]:
                                raise RuntimeError("Common-prefix frame mismatch")
                after = state_snapshot(task)
                results[name] = {"before": before, "after": after,
                                 "prefix_match": match, "observation_signature": signature,
                                 "executed": executed, "observation_updates": 32,
                                 "executed_actions": proposal[name][:executed].tolist(),
                                 "success_after": bool(task.eval_success),
                                 "frames": frames, "entity_id_map_offline": mapping,
                                 "history_observation_available": any(f["offset"] < 0 for f in frames)}
                if extended:
                    end_images, end_command = unpack_observation(obs)
                    arrays = dict(proposed_actions=proposal[name],
                                  residual_actions=proposal[name] - base,
                                  planned_cartesian_residual=cartesian[name],
                                  history_actions=np.asarray(prefix[-args.record_stride:], dtype=np.float32).reshape(-1, 14),
                                  executed_mask=np.asarray(executed_mask, dtype=np.bool_),
                                  physics_dt_s=np.asarray(durations, dtype=np.float64),
                                  physics_steps=np.asarray(physics_steps, dtype=np.int64),
                                  command_state_after=end_command,
                                  head_rgb_after=end_images[0], right_rgb_after=end_images[1],
                                  left_rgb_after=end_images[2])
                    transition = validate_transition(arrays)
                    results[name]["windows"] = pulse_windows(frames, arrays["executed_mask"], arrays["physics_dt_s"], arrays["physics_steps"], args.pulse_start)
                    np.savez_compressed(run / f"transition_{name}.npz", **arrays)
                    results[name].update(transition=transition,
                                         articulation_limits_offline_only=joint_limits,
                                         after_observation_signature=observation_signature(obs))
                if name == "base":
                    baseline_match = compare_physical(rows[args.chunk]["after"], after, args.tolerance)
                    if not baseline_match["matched"] or executed != 32:
                        raise RuntimeError(f"Original L0 branch not reproduced: {baseline_match}")
                print(f"{name} branch complete: executed={executed}", flush=True)
            finally:
                if hasattr(task, "scene"):
                    task.close_env()
        report = {"status": "complete", "valid_same_state_intervention": True,
                  "schema_version": 4,
                  "record_stride": args.record_stride,
                  "proposal_set": args.proposal_set,
                  "task": task_name, "seed": episode["seed"], "chunk": args.chunk,
                  "source_observation_exact_match": source_observation_match,
                  "delta_max_abs": float(max(np.max(np.abs(v-base)) for v in proposal.values())),
                  "branches": results, "baseline_source_match": baseline_match}
        write_json(run / "result.json", report)
        if extended:
            from .audit_cartesian_pulses import audit_run
            write_json(run / "schema_audit.json", audit_run(run))
        print(json.dumps({k: v for k, v in report.items() if k != "branches"}), flush=True)
    except BaseException:
        write_json(run / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
