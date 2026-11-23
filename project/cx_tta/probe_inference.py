"""Bounded fixed-input inference audit; no simulator and no success-rate claims."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .paths import ROOT, new_run, write_json, output_path
from .audit import digest


def prepare():
    import h5py
    from .dataset import read_observation
    manifest = ROOT / 'outputs/base_cache_20260926_024808_872087/manifest.json'
    rows = json.loads(manifest.read_text())['files']
    run = new_run('fixed_inputs', group=('diagnostics', 'inference'))
    records = []
    for task in sorted({r['task'] for r in rows}):
        row = min((r for r in rows if r['task'] == task), key=lambda r:r['episode_seed'])
        assert digest(Path(row['source'])) == row['source_sha256']
        with h5py.File(row['source'], 'r') as h:
            for frame in (0, len(h['joint_action/vector']) // 2):
                images, state = read_observation(h, frame)
                name = f'{task}_{frame}.npz'
                np.savez_compressed(run / name, state=state, prompt=np.asarray(str(h.attrs['instruction'])),
                                    noise=np.random.default_rng(100380 + frame).standard_normal((32,32)).astype(np.float32),
                                    **{f'image_{i}':v for i,v in enumerate(images)})
                records.append({'path':name, 'sha256':digest(run/name), 'task':task, 'frame':frame,
                                'source':row['source'], 'source_sha256':row['source_sha256']})
    write_json(run/'inputs.json', {'selection':'first collected episode by seed per task, first and middle frame; chosen without policy outcomes', 'records':records})
    print(run, flush=True)


def worker(args):
    from .evaluate_b2 import configure_runtime, resolve_tokenizer, seed_all, write_provenance
    from .cache_expert import create_base_policy
    from .eval_diagnostics import array_hash
    run = new_run('probe', group=('diagnostics', 'inference'))
    print(run, flush=True)
    bundle = args.bundle.resolve()
    records = json.loads((bundle/'inputs.json').read_text())['records']
    write_json(run/'manifest.json', {'mode':args.mode, 'bundle':str(bundle), 'input_manifest_sha256':digest(bundle/'inputs.json'), 'repeats':2, 'gpu':args.gpu})
    try:
        runtime=run/'runtime';runtime.mkdir()
        configure_runtime(run, runtime, resolve_tokenizer())
        import torch
        seed_all(0)
        # Constructor wraps sample_actions with torch.compile. Disabling Dynamo before
        # construction leaves the original numerical implementation in this process.
        if args.mode == 'eager':
            torch._dynamo.config.disable = True
        policy=create_base_policy()
        write_provenance(run, runtime, resolve_tokenizer())
        write_json(run/'hardware.json', {'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__,
            'cuda':torch.version.cuda, 'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'deterministic':torch.are_deterministic_algorithms_enabled(), 'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'dynamo_disabled':torch._dynamo.config.disable,
            'parameter_dtypes':sorted({str(v.dtype) for v in policy._model.parameters()})})
        captured={}
        transform=policy._input_transform
        def wrapped_transform(obs):
            result=transform(obs)
            def hashes(x):
                return {k:hashes(v) for k,v in x.items()} if isinstance(x,dict) else array_hash(np.asarray(x))
            captured['transformed']=hashes(result)
            return result
        policy._input_transform=wrapped_transform
        results=[]
        for repeat in range(2):
            for i,record in enumerate(records):
                src=bundle/record['path'];assert digest(src)==record['sha256']
                with np.load(src,allow_pickle=False) as z:
                    obs={'state':z['state'].copy(),'prompt':str(z['prompt']),
                         'images':{k:z[f'image_{j}'].transpose(2,0,1).copy() for j,k in enumerate(('cam_high','cam_right_wrist','cam_left_wrist'))}}
                    noise=z['noise'].copy()
                seed_all(100380)
                rng_before={'cpu':array_hash(torch.get_rng_state().numpy()),'cuda':array_hash(torch.cuda.get_rng_state().cpu().numpy())}
                start=time.perf_counter()
                actions=np.asarray(policy.infer(obs,noise=noise)['actions'],np.float32)
                elapsed=time.perf_counter()-start
                assert actions.shape==(32,14) and np.isfinite(actions).all()
                rng_after={'cpu':array_hash(torch.get_rng_state().numpy()),'cuda':array_hash(torch.cuda.get_rng_state().cpu().numpy())}
                np.save(run/f'r{repeat}_input{i}.npy',actions)
                results.append({'repeat':repeat,'input':i,'actions_sha256':array_hash(actions),'transformed':captured['transformed'],
                                'rng_before':rng_before,'rng_after':rng_after,'seconds':elapsed})
                print(f'{args.mode} repeat={repeat} input={i} seconds={elapsed:.3f}',flush=True)
        write_json(run/'result.json',{'status':'complete','records':results})
    except Exception:
        import traceback
        write_json(run/'failure.json',{'traceback':traceback.format_exc()})
        raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','worker'])
    p.add_argument('--bundle',type=Path);p.add_argument('--mode',choices=['compiled','eager'],default='compiled');p.add_argument('--gpu',default='0')
    args=p.parse_args()
    if args.action=='prepare':prepare()
    else:
        if not args.bundle or not args.gpu.isdigit():p.error('worker requires bundle and numeric GPU')
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
        worker(args)
if __name__=='__main__':main()
