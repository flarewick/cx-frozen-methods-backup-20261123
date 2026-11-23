"""Cache frozen checkpoint vision-tower spatial features of INITIAL images only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .audit import digest
from .cache_expert import CHECKPOINT, RT
from .intervention_batch import DEFAULT_PLAN
from .object_effect_labels import labels_for_run
from .paths import ROOT, new_run, output_path, replace_json, write_json

PREFIX = 'paligemma_with_expert.paligemma.model.vision_tower.'


def setup_imports(run):
    for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TORCH_HOME', 'JAX_COMPILATION_CACHE_DIR'):
        p = run / 'cache' / key.lower()
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'  # Official uint8 resize only; Torch encodes on chosen device.
    sys.path.insert(0, str(RT / 'RoboTwin/policy/pi05/src'))


def load_encoder(device):
    from safetensors import safe_open
    from transformers import SiglipVisionModel
    from transformers.models.auto import CONFIG_MAPPING
    config = CONFIG_MAPPING['paligemma']().vision_config
    config.intermediate_size = 4304
    config.projection_dim = 2048
    config.projector_hidden_act = 'gelu_fast'
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = SiglipVisionModel(config)
    state, sha = {}, hashlib.sha256()
    with safe_open(str(CHECKPOINT / 'model.safetensors'), framework='pt', device='cpu') as source:
        for key in sorted(k for k in source.keys() if k.startswith(PREFIX)):
            tensor = source.get_tensor(key)
            sha.update(key.encode())
            sha.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            sha.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            state[key[len(PREFIX):]] = tensor
    model.load_state_dict(state, strict=True, assign=True)
    # position_ids is a deterministic nonpersistent buffer, absent from the
    # checkpoint; load_state_dict cannot materialize it from the meta device.
    for name, buffer in list(model.named_buffers()):
        if buffer.is_meta:
            if not name.endswith('.position_ids') or buffer.shape != (1, 256):
                raise ValueError('Unexpected meta buffer: ' + name)
            owner, key = name.rsplit('.', 1)
            model.get_submodule(owner).register_buffer(
                key, torch.arange(256, dtype=buffer.dtype)[None], persistent=False)
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model, {'vision_tensor_sha256': sha.hexdigest(), 'tensor_count': len(state),
                   'config': config.to_dict(), 'parameters': sum(p.numel() for p in model.parameters())}


def preprocess(images):
    from openpi.shared.image_tools import resize_with_pad
    prepared = [np.asarray(resize_with_pad(image, 224, 224)) for image in images]
    # Same uint8 padding/rounding then [-1,1] normalization as the policy transforms.
    return torch.from_numpy(np.stack(prepared).copy()).permute(0, 3, 1, 2).float() / 127.5 - 1


def encode(model, images, device):
    with torch.inference_mode():
        tokens = model(pixel_values=preprocess(images).to(device)).last_hidden_state.float().cpu()
    if tokens.shape != (3, 256, 1152) or not torch.isfinite(tokens).all():
        raise ValueError('Unexpected vision tokens')
    spatial = tokens.reshape(3, 16, 16, 1152)
    pooled = spatial.reshape(3, 4, 4, 4, 4, 1152).mean(dim=(2, 4))
    return tokens.numpy(), pooled.numpy()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.gpu.isdigit() or (args.limit is not None and args.limit < 1):
        parser.error('Invalid GPU or limit')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.device == 'cuda' else ''
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu --limit 1 for a bounded check')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run = output_path(args.output_dir) if args.output_dir else new_run('siglip', group=('feature_caches',))
    run.mkdir(parents=True, exist_ok=True)
    print(run, flush=True)
    setup_imports(run)
    import fcntl
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model, encoder = load_encoder(args.device)
        plan = json.loads(DEFAULT_PLAN.read_text())
        jobs = []
        for task in sorted({j['task'] for j in plan['jobs']}):
            path = ROOT / 'outputs/intervention_batches' / plan['batch_id'] / task / 'index.json'
            index = json.loads(path.read_text())
            if index['status'] != 'complete' or index['plan_sha256'] != digest(DEFAULT_PLAN):
                raise ValueError('Incomplete/mismatching source index')
            jobs.extend(index['records'])
        jobs.sort(key=lambda j: j['id'])
        expected = {j['id']: j for j in plan['jobs']}
        if len(jobs) != len(expected) or {j['id'] for j in jobs} != set(expected):
            raise ValueError('Missing/duplicate jobs')
        for job in jobs:
            if any(job[k] != expected[job['id']][k] for k in ('source', 'seed', 'chunk', 'split', 'task')):
                raise ValueError('Source split/identity changed')
        if args.limit:
            jobs = jobs[:args.limit]
        contract = {'encoder': encoder, 'checkpoint': str(CHECKPOINT), 'plan_sha256': digest(DEFAULT_PLAN),
                    'code_sha256': digest(Path(__file__)), 'labels_code_sha256': digest(Path(__file__).with_name('object_effect_labels.py')),
                    'device': args.device, 'dtype': 'float32', 'limit': args.limit,
                    'camera_order': ['head', 'right', 'left'], 'feature': 'pre-projector vision tower; no language conditioning',
                    'resize_code_sha256': digest(RT / 'RoboTwin/policy/pi05/src/openpi/shared/image_tools.py'),
                    'torch_version': torch.__version__}
        if (run / 'manifest.json').exists():
            if json.loads((run / 'manifest.json').read_text()) != contract:
                raise ValueError('Cache contract changed; use a new output directory')
        else:
            write_json(run / 'manifest.json', contract)
        files = []
        for job in jobs:
            source = Path(job['run'])
            source_manifest = json.loads((source / 'manifest.json').read_text())
            result = json.loads((source / 'result.json').read_text())
            if ((source / 'failure.json').exists() or result['status'] != 'complete'
                    or source_manifest['source_seed'] != job['seed']
                    or source_manifest['chunk'] != job['chunk']
                    or source_manifest['task'] != job['task']):
                raise ValueError('Invalid source run')
            checkpoint_hash = digest(source / 'checkpoint.npz')
            target = run / (job['id'] + '.npz')
            if target.exists():
                with np.load(target, allow_pickle=False) as data:
                    if (str(data['source_sha256']) != checkpoint_hash or data['tokens'].shape != (3, 256, 1152)
                            or not np.isfinite(data['tokens']).all()):
                        raise ValueError('Invalid existing feature cache')
            else:
                with np.load(source / 'checkpoint.npz', allow_pickle=False) as data:
                    images = [data[k + '_rgb'] for k in ('head', 'right', 'left')]
                    state, base = data['command_state'], data['base_actions']
                tokens, pooled = encode(model, images, args.device)
                with (run / (job['id'] + '.tmp')).open('wb') as stream:
                    np.savez_compressed(stream, tokens=tokens.astype(np.float16), pooled=pooled,
                                        command_state=state, base_actions=base, source_sha256=checkpoint_hash)
                (run / (job['id'] + '.tmp')).replace(target)
            label = run / (job['id'] + '_offline_labels.json')
            value = labels_for_run(result)
            if label.exists():
                if json.loads(label.read_text()) != value:
                    raise ValueError('Offline labels changed')
            else:
                write_json(label, value)
            files.append({'job': job, 'instruction': source_manifest['instruction'],
                          'features': str(target), 'features_sha256': digest(target),
                          'offline_labels': str(label), 'offline_labels_sha256': digest(label)})
            replace_json(run / 'progress.json', {'completed': len(files), 'total': len(jobs)})
            print(f"CACHED {job['id']} {len(files)}/{len(jobs)}", flush=True)
        write = replace_json if (run / 'result.json').exists() else write_json
        write(run / 'result.json', {'status': 'complete', 'limited_probe': args.limit is not None,
                                   'files': files, 'scope': 'frozen initial-image features and separate offline labels; not trained M1'})


if __name__ == '__main__':
    main()
