"""Shared B2 residual BC baseline. No direction/effect/TTA claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from .dataset import load_cache
from .residual import SharedResidualAdapter
from .paths import ROOT, new_run, write_json, replace_json
from .audit import digest


def read_training_data(roots, allow_probe=False):
    partitions = {"train": [], "validation": []}
    seen, checkpoint = set(), None
    sources = []
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete" or (manifest["limited_probe"] and not allow_probe):
            raise ValueError("Only complete, unlimited caches may train a reported model")
        identity = (manifest["checkpoint_sha256"], manifest["normalization_sha256"], manifest["context_kind"])
        if checkpoint is not None and checkpoint != identity:
            raise ValueError("Mixed base checkpoints/normalization/feature contracts")
        checkpoint = identity
        sources.append({"path": str(manifest_path), "sha256": digest(manifest_path)})
        for row in manifest["files"]:
            key = (row["task"], row["episode_seed"])
            if key in seen:
                raise ValueError(f"Episode leakage/duplicate: {key}")
            seen.add(key)
            arrays = load_cache(Path(row["path"]))
            if str(arrays["source_sha256"]) != row["source_sha256"]:
                raise ValueError("Source provenance mismatch")
            partitions[row["split"]].append((row["task"], arrays))
    return partitions, sources


def stack(rows):
    return [torch.from_numpy(np.concatenate([arrays[key] for _, arrays in rows]))
            for key in ("context", "base_action", "target", "valid")]


def objective(model, batch):
    context, base, target, valid = batch
    predicted, _ = model(context, base)
    error = ((predicted - target) / model.caps).square().mean(dim=-1)
    # Each sample has equal weight irrespective of tail padding.
    return ((error * valid).sum(dim=1) / valid.sum(dim=1)).mean()


def evaluate(model, rows, batch_size, device):
    model.eval()
    result = {}
    with torch.no_grad():
        for task in sorted({task for task, _ in rows}):
            dataset = TensorDataset(*stack([row for row in rows if row[0] == task]))
            total, n = 0.0, 0
            for batch in DataLoader(dataset, batch_size=batch_size):
                batch = [t.to(device) for t in batch]
                total += float(objective(model, batch)) * len(batch[0])
                n += len(batch[0])
            result[task] = total / n
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--caches", nargs="+", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr <= .1:
        parser.error("Invalid training budget")
    torch.manual_seed(args.seed)
    parts, provenance = read_training_data(args.caches, allow_probe=args.check_only)
    if not parts["train"]:
        raise ValueError("No train episodes")
    if not args.check_only and not parts["validation"]:
        raise ValueError("Need independent validation episodes; never split frames of one episode")
    if not args.check_only:
        expected = {t["name"] for t in json.loads((ROOT / "configs/train_four_tasks.json").read_text())["tasks"]}
        for split in ("train", "validation"):
            if {task for task, _ in parts[split]} != expected:
                raise ValueError(f"The registered four-task baseline requires all four tasks in {split}")
    caps = [.1] * 14
    caps[6] = caps[13] = .2
    model = SharedResidualAdapter(590, caps).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tensors = stack(parts["train"])
    out = (new_run("adapter_check", group=("reports", "adapter_checks"))
           if args.check_only else new_run("adapter_train"))
    print(out, flush=True)
    metadata = {"scope": "B2 shared residual BC, not direction/effect/TTA", "check_only": args.check_only,
                "source_manifests": provenance, "parameters": sum(p.numel() for p in model.parameters()),
                "caps_native_units": caps, "arguments": {**vars(args), "caches": list(map(str, args.caches))}}
    needed = (tensors[2] - tensors[1]).abs()
    valid_elements = tensors[3].unsqueeze(-1).expand_as(needed)
    metadata["expert_residual_exceeds_cap_fraction"] = float(
        ((needed > torch.tensor(caps)) & valid_elements).sum() / valid_elements.sum())
    write_json(out / "manifest.json", metadata)
    # Include every training task in the integration check, not just the first task's frames.
    tasks = sorted({task for task, _ in parts["train"]})
    per_task = max(1, args.batch_size // len(tasks))
    check_tensors = [stack([row for row in parts["train"] if row[0] == task]) for task in tasks]
    batch = [torch.cat([values[i][:per_task] for values in check_tensors]).to(args.device)
             for i in range(4)]
    original, delta = model(batch[0], batch[1])
    if not torch.equal(original, batch[1]) or delta.count_nonzero():
        raise AssertionError("Zero-initialized residual must preserve base policy")
    if args.check_only:
        validation = evaluate(model, parts["validation"], args.batch_size, args.device)
        model.train()
        loss = objective(model, batch)
        loss.backward()
        grad = float(model.head.weight.grad.norm())
        if not np.isfinite(grad) or grad <= 0:
            raise ValueError("No valid gradient on actual cached data")
        optimizer.step()
        with torch.no_grad():
            _, updated = model(batch[0], batch[1])
            if updated.count_nonzero() == 0 or not torch.isfinite(updated).all():
                raise ValueError("Optimizer update did not produce finite nonzero corrections")
        write_json(out / "check.json", {"loss_before": float(loss.detach()), "gradient_norm": grad,
                                      "samples": len(batch[0]), "tasks_checked": tasks,
                                      "zero_residual_validation_by_task": validation, "status": "passed",
                                      "trained_model_claim": False})
        return
    counts = Counter()
    for task, arrays in parts["train"]:
        counts[task] += len(arrays["context"])
    weights = torch.tensor([1.0 / counts[task] for task, arrays in parts["train"] for _ in arrays["context"]])
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size,
                        sampler=WeightedRandomSampler(weights, len(weights), replacement=True))
    baseline = evaluate(model, parts["validation"], args.batch_size, args.device)
    baseline_macro = float(np.mean(list(baseline.values())))
    write_json(out / "baseline.json", {"zero_residual_validation_by_task": baseline,
                                      "zero_residual_validation_macro": baseline_macro})
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = [t.to(args.device) for t in batch]
            optimizer.zero_grad()
            loss = objective(model, batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, parts["validation"], args.batch_size, args.device)
        score = float(np.mean(list(validation.values())))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro": score,
                        "validation_by_task": validation})
        replace_json(out / "history.json", history)
        print(f"epoch {epoch + 1}/{args.epochs}: validation_macro={score:.6f} baseline={baseline_macro:.6f}", flush=True)
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "metadata": metadata, "epoch": epoch,
                        "validation_macro": score}, out / "best.pt")
    write_json(out / "result.json", {"status": "complete", "epochs": args.epochs,
                                   "best_validation_macro": best, "zero_residual_validation_macro": baseline_macro,
                                   "beats_zero_residual_offline": best < baseline_macro,
                                   "robot_success_tested": False})
    print(f"Finished B2 training; validation loss {best:.6f}. Robot success untested.", flush=True)


if __name__ == "__main__":
    main()
