"""Read-only G0 inventory. No model, simulator, torch, or pickle import."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import struct
import sys
import zipfile
from .paths import ROOT, new_run, write_json

SKIP = {"__pycache__", ".git", "cache", ".cache", "site-packages", "node_modules"}
DATA_EXT = {".npz", ".h5", ".hdf5", ".parquet", ".tfrecord"}


def files(root):
    if not root.exists():
        return
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not Path(parent, d).is_symlink())
        for name in sorted(names):
            path = Path(parent, name)
            if not path.is_symlink():
                yield path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def npz_headers(path):
    """Read only NPY headers. Never unpickle legacy object arrays."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with archive.open(info) as stream:
                if stream.read(6) != b"\x93NUMPY":
                    raise ValueError("Invalid NPY magic")
                version = tuple(stream.read(2))
                if version not in {(1, 0), (2, 0), (3, 0)}:
                    raise ValueError(f"Unsupported NPY header version {version}")
                width = 2 if version == (1, 0) else 4
                length = struct.unpack("<H" if width == 2 else "<I", stream.read(width))[0]
                if length > 65536:
                    raise ValueError("NPY header exceeds inspection budget")
                header = ast.literal_eval(stream.read(length).decode("utf-8" if version == (3, 0) else "latin1"))
                result[info.filename[:-4]] = {
                    "shape": list(header["shape"]), "dtype": header["descr"],
                    "array_bytes": info.file_size, "data_loaded": False,
                }
    return result


def infer_task(path, tasks):
    matches = [t for t in tasks if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", str(path))]
    return max(matches, key=len) if matches else None


def episode_inventory(path, tasks):
    """Preserve duplicates/conflicts; never deduplicate into success claims."""
    h, seen = hashlib.sha256(), set()
    duplicates, errors, rows = [], [], []
    with path.open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            h.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                task = row.get("task") or row.get("task_name") or infer_task(path, tasks)
                seed = row.get("seed")
                if not task or type(seed) is not int:
                    raise ValueError("Missing task or integer seed")
                condition = row.get("cond", row.get("condition", "unknown"))
                key = (task, seed, condition)
                if key in seen:
                    duplicates.append({"task": task, "seed": seed, "condition": condition, "line": line_no})
                seen.add(key)
                rows.append({"task": task, "seed": seed, "condition": condition,
                             "task_config": row.get("task_config", "unknown"),
                             "success": row.get("success"), "instruction": row.get("instruction"),
                             "line": line_no})
            except (ValueError, TypeError, AttributeError) as exc:
                errors.append({"line": line_no, "error": str(exc)})
    return {"path": str(path), "sha256": h.hexdigest(), "rows": rows,
            "duplicates": duplicates, "errors": errors}


def verify_legacy(manifest):
    return [name for name, expected in manifest.items()
            if not Path(name).is_file() or digest(Path(name)) != expected]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/audit.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = new_run("g0_audit", group=("reports", "audits"))
    legacy = Path(config["legacy_root"])
    envs = Path(config["robotwin_root"]) / "envs"
    tasks = sorted(p.stem for p in envs.glob("*.py") if not p.name.startswith("_"))
    protected = json.loads((ROOT / "docs/legacy_source_hashes.json").read_text())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable, "gpu_used": False,
        "config": config, "config_sha256": digest(args.config), "task_registry": tasks,
        "packages": {}, "training_ready": False, "sources": [], "errors": [],
        "stage": "G0 inventory only; no dataset/split/action contract validation",
        "legacy_changed_before": verify_legacy(protected),
    }
    for package in ("numpy", "torch", "jax", "sapien", "h5py", "transformers"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    task_usage = defaultdict(lambda: {"seeds": set(), "conditions": Counter(), "files": set()})
    episode_files, dataset_samples = [], {}
    for source in config["data_roots"]:
        root, counts = Path(source), Counter()
        source_info = {"path": str(root), "exists": root.exists(), "data_candidates": [],
                       "symlink_directories": [], "zarr_directories": []}
        if root.exists():
            for parent, dirs, _ in os.walk(root, followlinks=False):
                dirs[:] = [d for d in dirs if d not in SKIP]
                for name in dirs:
                    if Path(parent, name).is_symlink():
                        source_info["symlink_directories"].append(str(Path(parent, name)))
                    if name.endswith(".zarr"):
                        source_info["zarr_directories"].append(str(Path(parent, name)))
        for path in files(root):
            counts[path.suffix.lower() or "(none)"] += 1
            if path.name in {"episodes.jsonl", "episodes_v3.jsonl"}:
                try:
                    item = episode_inventory(path, tasks)
                    episode_files.append(item)
                    for row in item["rows"]:
                        usage = task_usage[row["task"]]
                        usage["seeds"].add(row["seed"])
                        usage["conditions"][str(row["condition"])] += 1
                        usage["files"].add(str(path))
                except OSError as exc:
                    report["errors"].append(f"{path}: {exc}")
            if path.suffix.lower() in DATA_EXT or ".tfrecord-" in path.name:
                task = infer_task(path, tasks)
                relative = path.relative_to(root)
                suite = relative.parts[0] if len(relative.parts) > 1 else "."
                source_info["data_candidates"].append({
                    "path": str(path), "bytes": path.stat().st_size, "inferred_task": task,
                })
                key = str(path)
                if path.suffix == ".npz" and key not in dataset_samples:
                    try:
                        dataset_samples[key] = {"path": str(path), "inferred_task": task,
                                                "headers": npz_headers(path)}
                    except (ValueError, OSError, zipfile.BadZipFile, SyntaxError) as exc:
                        dataset_samples[key] = {"path": str(path), "error": str(exc)}
        source_info["file_counts"] = dict(counts)
        report["sources"].append(source_info)
    report["npz_schema_samples"] = list(dataset_samples.values())
    report["schema_sampling"] = "All discovered NPZ headers; contents and temporal alignment NOT validated"
    report["training_sources"] = {
        name: {"exists": (legacy / name).exists()}
        for name in ("models/residual_v4.py", "tta/tta_v4.py", "tta/tta_v5.py",
                     "tta/action_adapter.py", "training/train_shared_residual.py")
    }
    report["checkpoint_inventory"] = [
        {"path": str(p), "bytes": p.stat().st_size,
         "sha256": digest(p) if p.stat().st_size <= 16 * 1024 * 1024 else None}
        for p in files(Path(config["checkpoint_root"]))
    ]
    report["checkpoint_hash_policy"] = "Files <=16MiB hashed; large weights metadata only"
    report["legacy_changed_after"] = verify_legacy(protected)
    usage = {task: {"seeds": sorted(v["seeds"]), "conditions": dict(v["conditions"]),
                    "files": sorted(v["files"])} for task, v in sorted(task_usage.items())}
    report["episode_parse_errors"] = sum(len(i["errors"]) for i in episode_files)
    report["duplicate_episode_keys"] = sum(len(i["duplicates"]) for i in episode_files)
    write_json(out / "episode_index.json", episode_files)
    write_json(out / "used_seeds.json", usage)
    write_json(out / "audit.json", report)
    lines = ["# G0 只读审计", "", "本报告不启动 GPU，不代表训练数据或任务划分已通过验收。", "",
             f"- 本地任务源码数：{len(tasks)}",
             f"- episode 日志数：{len(episode_files)}；有历史记录任务数：{len(usage)}",
             f"- NPZ schema 样本数：{len(dataset_samples)}（只读数组头，不加载 pickle）",
             f"- 受保护旧文件变化：{len(report['legacy_changed_after'])}",
             f"- 日志解析问题：{report['episode_parse_errors']}；文件内重复键：{report['duplicate_episode_keys']}",
             "", "## 扫描的数据根", "", "|路径|存在|候选数据文件数|", "|---|---|---:|"]
    lines += [f"|{s['path']}|{s['exists']}|{len(s['data_candidates'])}|" for s in report["sources"]]
    lines += ["", "## 历史使用情况", "", "|任务|使用过的 seed 数|条件记录数（含重复运行）|",
              "|---|---:|---|"]
    lines += [f"|{task}|{len(v['seeds'])}|{v['conditions']}|" for task, v in usage.items()]
    lines += ["", "## 尚未通过的项目", "",
              "- 完整示范/恢复动作与观测转移的来源、同步、合法监督尚需验证。",
              "- schema 只作线索，不能把旧反事实 NPZ 直接当成有效纠错标签。",
              "- seed 清单只覆盖扫描的 episode 日志；须结合历史文档和其他日志预注册。",
              "- 训练/验证/留出任务尚未冻结；动作语义与部署加载链未通过集成验收。",
              "- 大权重未完整哈希，不能宣称基础 checkpoint 身份已锁定。",
              "", "详细 schema、文件来源和错误见 audit.json；逐行索引见 episode_index.json。"]
    (out / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out, flush=True)
    print(json.dumps({"tasks": len(tasks), "episode_files": len(episode_files),
                      "historical_tasks": len(usage), "npz_schemas": len(dataset_samples),
                      "legacy_changed": report["legacy_changed_after"], "training_ready": False}))
    return int(bool(report["legacy_changed_before"] or report["legacy_changed_after"] or report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
