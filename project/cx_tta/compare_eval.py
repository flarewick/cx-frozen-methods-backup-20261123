"""Compare completed, exactly paired RoboTwin L0/B2 episode results."""
import argparse
import json
from pathlib import Path
from .paths import new_run, write_json
from .audit import digest


def compare(l0, b2):
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (l0, b2)]
    results = [json.loads((root / "result.json").read_text()) for root in (l0, b2)]
    if any(row["status"] != "complete" for row in results):
        raise ValueError("Both evaluations must be complete")
    a, b = (row["args"] for row in manifests)
    if (a["condition"], b["condition"]) != ("L0", "B2"):
        raise ValueError("Expected L0 then B2")
    for field in ("task", "task_config", "instruction_type", "seed_start", "episodes"):
        if a[field] != b[field]:
            raise ValueError(f"Different evaluation contract: {field}")
    for field, default in (("inference_mode", "compiled"), ("policy_init_seed", None),
                           ("rollout_protocol", "paired")):
        if a.get(field, default) != b.get(field, default):
            raise ValueError(f"Different inference contract: {field}")
    if manifests[0]["seeds"] != manifests[1]["seeds"]:
        raise ValueError("Seed sets differ")
    if manifests[0].get("checkpoint_sha256") != manifests[1].get("checkpoint_sha256"):
        raise ValueError("Frozen checkpoint hashes differ")
    if manifests[0].get("legacy_l0_sha256") != manifests[1].get("legacy_l0_sha256"):
        raise ValueError("Historical seed/instruction source hashes differ")
    if (l0 / "runtime_hashes.json").read_text() != (b2 / "runtime_hashes.json").read_text():
        raise ValueError("Simulator source or configuration differs")
    episodes = [json.loads((root / "episodes.json").read_text()) for root in (l0, b2)]
    if [x["seed"] for x in episodes[0]] != [x["seed"] for x in episodes[1]]:
        raise ValueError("Executed seed order differs")
    paired = []
    for x, y in zip(*episodes):
        if x["eligible"] != y["eligible"]:
            raise ValueError(f"Expert eligibility mismatch on seed {x['seed']}")
        if not x["eligible"]:
            continue
        if x["instruction"] != y["instruction"]:
            raise ValueError(f"Instructions differ on seed {x['seed']}")
        paired.append({"seed": x["seed"], "l0_success": x["success"], "b2_success": y["success"],
                       "l0_actions": x["executed_actions"], "b2_actions": y["executed_actions"],
                       "b2_max_abs_correction": y["max_abs_correction"]})
    if not paired:
        raise ValueError("No eligible paired scenes")
    return {"task": a["task"], "status": "paired", "attempted": len(episodes[0]),
            "eligible": len(paired), "l0_success": sum(x["l0_success"] for x in paired),
            "b2_success": sum(x["b2_success"] for x in paired),
            "b2_only": sum(x["b2_success"] and not x["l0_success"] for x in paired),
            "l0_only": sum(x["l0_success"] and not x["b2_success"] for x in paired),
            "both_success": sum(x["l0_success"] and x["b2_success"] for x in paired),
            "neither_success": sum(not x["l0_success"] and not x["b2_success"] for x in paired),
            "episodes": paired, "sources": {"L0": str(l0.resolve()), "B2": str(b2.resolve())},
            "source_sha256": {"L0": digest(l0 / "episodes.json"),
                              "B2": digest(b2 / "episodes.json")}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--l0", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.l0, args.b2)
    out = new_run("paired_eval", group=("evaluations", result["task"]))
    write_json(out / "report.json", result)
    print(out)
    print(json.dumps({k: result[k] for k in ("attempted", "eligible", "l0_success", "b2_success",
                                            "b2_only", "l0_only")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
