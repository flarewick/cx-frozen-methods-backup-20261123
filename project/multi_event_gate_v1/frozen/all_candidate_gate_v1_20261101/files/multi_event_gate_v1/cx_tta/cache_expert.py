"""Query the frozen VLA on the EXACT demonstration observations, not separate rollouts."""
import argparse
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
from .audit import digest
from .paths import ROOT, new_run, write_json, replace_json

RT = Path("/data/cj/robo_pi/250010033/qfw/robotwin")
CHECKPOINT = RT / "checkpoints/pi05_motus_robotwin2/motus_robotwin2/1000000"


def create_base_policy():
    sys.path.insert(0, str(RT / "RoboTwin/policy/pi05/src"))
    sys.path.insert(0, str(RT / "pi05_integration"))
    import register_motus_config
    from openpi.policies import policy_config
    from openpi.training import config, checkpoints
    norms = checkpoints.load_norm_stats(CHECKPOINT / "assets", "pi0.5_clean_randomize_joint_training")
    return policy_config.create_trained_policy(config.get_config("pi05_motus_robotwin2"),
                                               CHECKPOINT, norm_stats=norms, pytorch_device="cuda")


def cache_episode(policy, source, target, *, stride=8, max_frames=None, noise_seed=0):
    import h5py
    import numpy as np
    from .capture import validate_capture
    from .dataset import CONTEXT_KIND, read_observation, context_features, expert_chunk
    validate_capture(source, decode_all=False)
    rows = []
    with h5py.File(source, "r") as h:
        commands = h["joint_action/vector"][:]
        prompt = str(h.attrs["instruction"])
        indexes = list(range(0, len(commands) - 1, stride))
        if max_frames is not None:
            indexes = indexes[:max_frames]
        rng = np.random.default_rng(noise_seed)
        for index in indexes:
            images, state = read_observation(h, index)
            observation = {
                "state": state, "prompt": prompt,
                "images": {name: image.transpose(2, 0, 1)
                           for name, image in zip(("cam_high", "cam_right_wrist", "cam_left_wrist"), images)},
            }
            # Explicit per-episode RNG; exactly one query per observation, no best-of-N.
            noise = rng.standard_normal((32, 32)).astype(np.float32)
            base = np.asarray(policy.infer(observation, noise=noise)["actions"], dtype=np.float32)
            if base.shape != (32, 14):
                raise ValueError(f"Unexpected native policy output: {base.shape}")
            expert, valid = expert_chunk(commands, index)
            rows.append((index, context_features(images, state), base, expert, valid))
            print(f"cached frame {index} / {len(commands)}", flush=True)
        np.savez_compressed(target, frame_index=np.asarray([r[0] for r in rows]),
                            context=np.stack([r[1] for r in rows]), base_action=np.stack([r[2] for r in rows]),
                            target=np.stack([r[3] for r in rows]), valid=np.stack([r[4] for r in rows]),
                            context_kind=np.asarray(CONTEXT_KIND), source_sha256=np.asarray(digest(source)),
                            task=np.asarray(h.attrs["task"]), episode_seed=np.asarray(h.attrs["seed"]))
        return {"path": str(target), "source": str(source), "source_sha256": digest(source),
                "samples": len(rows), "task": str(h.attrs["task"]), "episode_seed": int(h.attrs["seed"]),
                "instruction": prompt, "noise_seed": noise_seed}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--tokenizer-path", type=Path,
                        default=Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model")
    args = parser.parse_args()
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("positive stride/frame count required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = new_run("base_cache")
    for key in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TMPDIR",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                "JAX_COMPILATION_CACHE_DIR", "NUMBA_CACHE_DIR"):
        folder = out / "cache" / key.lower()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    tempfile.tempdir = os.environ["TMPDIR"]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OPENPI_DATA_HOME"] = str(out / "cache/openpi")
    token_target = out / "cache/openpi/big_vision/paligemma_tokenizer.model"
    token_target.parent.mkdir(parents=True)
    if not args.tokenizer_path.is_file():
        raise FileNotFoundError("Pass --tokenizer-path to an existing local Paligemma tokenizer")
    shutil.copyfile(args.tokenizer_path, token_target)
    print(out, flush=True)
    manifest = {"schema_version": 1, "checkpoint": str(CHECKPOINT),
                "checkpoint_sha256": digest(CHECKPOINT / "model.safetensors"),
                "normalization_sha256": digest(CHECKPOINT / "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"),
                "tokenizer_sha256": digest(token_target),
                "label_alignment": "state/image at i, expert absolute commands i+1 ... i+32",
                "context_kind": "native_rgb_grid8_command_state14_v1", "files": [],
                "status": "running", "limited_probe": args.max_frames is not None}
    try:
        policy = create_base_policy()
        keys = set()
        for collection in args.collections:
            result = json.loads((collection / "result.json").read_text())
            if result["status"] not in {"collection_complete", "format_probe_complete"}:
                raise ValueError("Collection incomplete; audit partial data explicitly before training")
            collect_manifest = json.loads((collection / "manifest.json").read_text())
            for row in json.loads((collection / "attempts.json").read_text()):
                if not row["replay_success"] or not row["format_valid"]:
                    continue
                task = collect_manifest["arguments"]["task"]
                key = (task, row["seed"])
                if key in keys:
                    raise ValueError(f"Duplicate task/seed across collections: {key}")
                keys.add(key)
                metadata = cache_episode(policy, Path(row["hdf5"]), out / f"{task}_{row['seed']}.npz",
                                         stride=args.stride, max_frames=args.max_frames, noise_seed=row["seed"])
                metadata["split"] = row["split"]
                metadata["collection"] = str(collection.resolve())
                manifest["files"].append(metadata)
                replace_json(out / "manifest.json", manifest)
        if not manifest["files"]:
            raise ValueError("No validated successful demonstrations")
        manifest["status"] = "complete"
        replace_json(out / "manifest.json", manifest)
    except BaseException:
        manifest["status"] = "failed_or_interrupted"
        replace_json(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
