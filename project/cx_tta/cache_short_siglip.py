"""Frozen semantic cache: history and initial RGB only, 25 train checkpoints."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .cache_siglip import setup_imports,load_encoder,encode,CHECKPOINT
from .paths import new_run,output_path,write_json,replace_json
from .short_window_dataset import load_jobs


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--gpu',default='0');p.add_argument('--limit',type=int);p.add_argument('--output-dir',type=Path);args=p.parse_args()
    if not args.gpu.isdigit() or (args.limit is not None and args.limit<1):p.error('Invalid GPU/limit')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu if args.device=='cuda' else ''
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    run=output_path(args.output_dir) if args.output_dir else new_run('short_siglip',group=('feature_caches',));run.mkdir(parents=True,exist_ok=True);print(run,flush=True)
    setup_imports(run)
    import fcntl
    with (run/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=sorted(load_jobs(),key=lambda j:j['id']);jobs=jobs[:args.limit] if args.limit else jobs
        model,encoder=load_encoder(args.device)
        contract=dict(encoder=encoder,checkpoint=str(CHECKPOINT),device=args.device,limit=args.limit,
                      code_sha256=digest(Path(__file__)),encoder_code_sha256=digest(Path(__file__).with_name('cache_siglip.py')),
                      jobs=[j['id'] for j in jobs],scope='Only pre-intervention offsets -8 and 0; pooled 3x4x4x1152; no GT/future pixels or language conditioning')
        if (run/'manifest.json').exists():
            if json.loads((run/'manifest.json').read_text())!=contract:raise ValueError('Cache contract differs')
        else:write_json(run/'manifest.json',contract)
        records=[]
        for job in jobs:
            source=Path(job['run']);result=json.loads((source/'result.json').read_text())
            if result['status']!='complete' or (source/'failure.json').exists():raise ValueError('Invalid run')
            frames={f['offset']:f for f in result['branches']['base']['frames']}
            for offset in (-8,0):
                frame=frames[offset];src=source/frame['observed_file'];sha=digest(src)
                if sha!=frame['observed_sha256']:raise ValueError('Source frame changed')
                dest=run/f"{job['id']}_{offset:+03d}.npz"
                if dest.exists():
                    with np.load(dest) as z:
                        if str(z['source_sha256'])!=sha or z['pooled'].shape!=(3,4,4,1152) or not np.isfinite(z['pooled']).all():raise ValueError('Invalid existing cache')
                else:
                    with np.load(src) as z:images=[z[k+'_rgb'] for k in ('head','right','left')]
                    _,pooled=encode(model,images,args.device)
                    tmp=dest.with_suffix('.tmp')
                    with tmp.open('wb') as stream:np.savez_compressed(stream,pooled=pooled,source_sha256=sha)
                    tmp.replace(dest)
                records.append(dict(job=job['id'],offset=offset,file=str(dest),sha256=digest(dest),source=str(src),source_sha256=sha))
            replace_json(run/'progress.json',{'completed_frames':len(records),'total_frames':len(jobs)*2});print(f"CACHED {job['id']} {len(records)}/{len(jobs)*2}",flush=True)
        replace_json(run/'result.json',dict(status='complete',limited=args.limit is not None,records=records))
        print(f'COMPLETE frames={len(records)} images={len(records)*3} output={run}',flush=True)

if __name__=='__main__':main()
