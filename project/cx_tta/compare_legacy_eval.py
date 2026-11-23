"""Pair a new B2 run with the audited historical L0 seed and instruction rows."""
import argparse
import json
from pathlib import Path
from .audit import digest
from .evaluate_b2 import read_legacy_l0
from .paths import new_run, write_json


def compare(legacy_l0, b2):
    manifest = json.loads((b2 / "manifest.json").read_text())
    result = json.loads((b2 / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("B2 evaluation incomplete")
    args = manifest["args"]
    if args["condition"] != "B2" or Path(args["legacy_l0"]).resolve() != legacy_l0.resolve():
        raise ValueError("B2 run does not reference this historical baseline")
    if manifest["legacy_l0_sha256"] != digest(legacy_l0):
        raise ValueError("Historical L0 changed since B2 evaluation")
    old = read_legacy_l0(legacy_l0, args["task"])
    rows = json.loads((b2 / "episodes.json").read_text())
    if [row["seed"] for row in rows] != list(old):
        raise ValueError("B2 seed order differs from historical L0")
    paired = []
    for row in rows:
        ref = old[row["seed"]]
        if not row["eligible"] or row["instruction"] != ref["instruction"]:
            raise ValueError(f"Seed/instruction mismatch: {row['seed']}")
        paired.append({"seed": row["seed"], "l0_success": ref["success"],
                       "b2_success": row["success"], "instruction": row["instruction"],
                       "b2_max_abs_correction": row["max_abs_correction"]})
    return {"status": "paired_historical_diagnostic", "task": args["task"],
            "n": len(paired), "l0_success": sum(x["l0_success"] for x in paired),
            "b2_success": sum(x["b2_success"] for x in paired),
            "b2_only": sum(x["b2_success"] and not x["l0_success"] for x in paired),
            "l0_only": sum(x["l0_success"] and not x["b2_success"] for x in paired),
            "episodes": paired, "legacy_l0": str(legacy_l0.resolve()),
            "legacy_l0_sha256": digest(legacy_l0), "b2_run": str(b2.resolve()),
            "comparability_limit": "Historical simulator and policy bytecode hashes are not fully recorded; seed and instruction are exact but bitwise environmental equivalence is unproven."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--legacy-l0", type=Path, required=True)
    parser.add_argument("--b2", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.legacy_l0, args.b2)
    out = new_run("legacy_pair", group=("evaluations", report["task"]))
    write_json(out / "report.json", report)
    print(out)
    print(json.dumps({k: report[k] for k in ("n", "l0_success", "b2_success", "b2_only", "l0_only")}))


if __name__ == "__main__":
    main()
