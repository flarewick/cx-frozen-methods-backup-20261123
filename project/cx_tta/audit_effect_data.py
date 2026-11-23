"""Read-only action/effect supervision audit of the frozen four-task HDF5 sources."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import h5py
import numpy as np
from .audit import digest
from .paths import ROOT, new_run, write_json

# Offline audit labels only; never used by a deployed router or policy.
PRIMARY_OBJECT = {"beat_block_hammer": "hammer", "place_shoe": "shoe",
                  "turn_switch": "switch", "place_empty_cup": "cup"}


def audit_episode(row):
    path = Path(row["source"])
    with h5py.File(path, "r") as h:
        ticks = h["cx_measurement/physics_tick"][:]
        dt = h["cx_measurement/physics_dt_s"][:]
        dense_ticks = h["cx_dense/tick_before"][:]
        dense_dt = h["cx_dense/dt_s"][:]
        commands = h["cx_dense/command"][:]
        arms = h["cx_measurement/arm_qpos"][:]
        delta_tick = np.diff(ticks)
        if (np.any(delta_tick < 0) or np.any(np.diff(dense_ticks) != 1)
                or commands.shape != (len(dense_ticks), 14)
                or not np.isfinite(commands).all() or not np.isfinite(arms).all()
                or np.any(dt <= 0) or np.any(dense_dt <= 0)):
            raise ValueError(f"Invalid physics timeline: {path}")
        if dense_ticks[0] > ticks[0] or dense_ticks[-1] + 1 < ticks[-1]:
            raise ValueError(f"Dense commands do not cover observations: {path}")
        # Integrate actual dt, without equating one sparse frame with one action.
        elapsed = np.concatenate(([0.0], np.cumsum(dense_dt)))
        intervals = np.diff(elapsed[ticks - dense_ticks[0]])
        actor = PRIMARY_OBJECT[row["task"]]
        pose = np.asarray(h[f"cx_privileged/{actor}"][:], dtype=np.float64)
        if pose.shape != (len(ticks), 7) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid actor poses: {path}")
        offset = np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1)
        q = pose[:, 3:] / np.linalg.norm(pose[:, 3:], axis=1, keepdims=True)
        rotation = 2 * np.arccos(np.clip(np.abs(q @ q[0]), 0, 1))
        positive = intervals[intervals > 0]
        return {"task": row["task"], "seed": row["episode_seed"], "split": row["split"],
                "source": str(path), "frames": len(ticks), "positive_time_transitions": int(np.sum(delta_tick > 0)),
                "zero_time_transitions": int(np.sum(delta_tick == 0)),
                "interval_s_min": float(positive.min()) if len(positive) else None,
                "interval_s_max": float(positive.max()) if len(positive) else None,
                "root_translation_max_m": float(offset.max()), "root_rotation_max_rad": float(rotation.max()),
                "articulation_qpos_recorded": "cx_privileged_articulation" in h,
                "source_sha256_matches_cache": digest(path) == row["source_sha256"]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/base_cache_20260926_024808_872087")
    args = parser.parse_args()
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["limited_probe"]:
        raise ValueError("Require complete formal cache source manifest")
    rows = [audit_episode(row) for row in manifest["files"]]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    stats = {}
    for task, values in grouped.items():
        stats[task] = {"episodes": len(values),
                       "frames": sum(r["frames"] for r in values),
                       "positive_time_transitions": sum(r["positive_time_transitions"] for r in values),
                       "zero_time_transitions": sum(r["zero_time_transitions"] for r in values),
                       "translation_max_median_m": float(np.median([r["root_translation_max_m"] for r in values])),
                       "rotation_max_median_rad": float(np.median([r["root_rotation_max_rad"] for r in values])),
                       "stationary_root_episodes": sum(r["root_translation_max_m"] < 1e-6 and r["root_rotation_max_rad"] < 1e-6 for r in values),
                       "with_articulation_qpos": sum(r["articulation_qpos_recorded"] for r in values)}
    out = new_run("effect_data_audit", group=("reports", "audits"))
    report = {"status": "audited" if all(r["source_sha256_matches_cache"] for r in rows) else "source_changed",
              "source_manifest_sha256": digest(args.cache / "manifest.json"),
              "tasks": stats, "episodes": rows,
              "limitations": ["Successful expert trajectories only; no causal residual-effect labels.",
                              "Switch success is articulated joint motion; root pose alone cannot supervise it.",
                              "GT actor poses are offline-only, not deployable model inputs."]}
    write_json(out / "report.json", report)
    print(out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if report["status"] != "audited":
        raise SystemExit("Source hashes differ")


if __name__ == "__main__":
    main()
