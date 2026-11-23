"""Inspect provided TFDS metadata and legacy NPZ schemas without model imports."""
import argparse
import json
from pathlib import Path
from .audit import digest, files, npz_headers
from .paths import new_run, write_json


def flatten_features(value, prefix=""):
    result = {}
    if not isinstance(value, dict):
        return result
    if "featuresDict" in value:
        for name, child in value["featuresDict"]["features"].items():
            result.update(flatten_features(child, f"{prefix}/{name}".strip("/")))
    elif "sequence" in value:
        result.update(flatten_features(value["sequence"]["feature"], prefix))
    elif "tensor" in value or "image" in value:
        field = value.get("tensor", value.get("image"))
        result[prefix] = {"shape": field.get("shape", {}).get("dimensions", []),
                          "dtype": field.get("dtype"), "description": value.get("description")}
    elif "text" in value:
        result[prefix] = {"type": "text"}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/data/cj/robo_pi/250010033/VLA-Adapter/data"))
    parser.add_argument("--legacy-npz-root", type=Path,
                        default=Path("/data/cj/robo_pi/250010033/VLA-Adapter/direction_guide_TTA_v5/outputs"))
    args = parser.parse_args()
    out = new_run("data_review", group=("reports", "audits"))
    datasets = []
    all_files = list(files(args.data_root))
    for path in all_files:
        if path.name != "features.json":
            continue
        info_path = path.with_name("dataset_info.json")
        datasets.append({
            "path": str(path.parent), "features_sha256": digest(path),
            "features": flatten_features(json.loads(path.read_text())),
            "dataset_info": json.loads(info_path.read_text()) if info_path.exists() else None,
            "shards": [{"path": str(p), "bytes": p.stat().st_size}
                       for p in all_files if p.parent == path.parent and ".tfrecord-" in p.name],
        })
    npzs = []
    for path in files(args.legacy_npz_root):
        if path.suffix == ".npz":
            npzs.append({"path": str(path), "headers": npz_headers(path)})
    report = {"provided_root": str(args.data_root), "provided_root_exists": args.data_root.exists(),
              "datasets": datasets, "npz_schemas": npzs,
              "hdf5_files": [str(p) for p in all_files if p.suffix in {".h5", ".hdf5"}],
              "training_ready": False, "inspection": "metadata only; no shard integrity validation",
              "limitation": "Need matching RoboTwin observations/native action convention and synchronized executed transitions"}
    write_json(out / "data_review.json", report)
    lines = ["# 现有数据适配性检查", "", "只读 metadata/NPZ 数组头；未载入训练样本，未启动 GPU。", "",
             "|数据目录|声明动作维度/语义|TFRecord 文件数|", "|---|---|---:|"]
    for d in datasets:
        action = d["features"].get("steps/action", {})
        lines.append(f"|{d['path']}|{action.get('shape')} / {action.get('description')}|{len(d['shards'])}|")
    lines += ["", f"- HDF5 文件数：{len(report['hdf5_files'])}。",
              f"- 旧 NPZ 文件数：{len(npzs)}，详细字段逐文件见 JSON。",
              "", "## 使用边界", "",
              "LIBERO 的 7 维末端动作不能通过补零或直接相减作为 RoboTwin 双臂关节残差标签。",
              "该数据可另作 LIBERO 实验或明确设计的视觉预训练，但当前 RoboTwin 主线需要匹配的示范/执行转移。",
              "旧 NPZ 中的候选方向、成功分数和局部图像不能替代同步原生动作与后续观测；未经快照一致性验证的分支分数不能作为因果标签。",
              "先采集少量 RoboTwin 数据验证格式、控制时间和路径，再扩大采集；不得直接启动训练并宣称已经满足 G0。"]
    (out / "data_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps({"datasets": len(datasets), "hdf5": len(report["hdf5_files"]), "npz": len(npzs),
                      "actions": [d["features"].get("steps/action") for d in datasets]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
