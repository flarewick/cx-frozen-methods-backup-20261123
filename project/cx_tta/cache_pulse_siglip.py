"""Frozen RGB tower at the actual pulse start and its history, never future frames."""
import argparse,json,os,fcntl
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .cache_siglip import setup_imports,load_encoder,encode,CHECKPOINT
from .paths import new_run,output_path,write_json,replace_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--gpu',default='0');p.add_argument('--output-dir',type=Path);p.add_argument('--limit',type=int);a=p.parse_args()
    if not a.gpu.isdigit() or (a.limit is not None and a.limit<1):p.error('Invalid GPU/limit')
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu if a.device=='cuda' else ''
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    index=json.loads((a.dataset/'index.json').read_text());checked=json.loads((a.dataset/'interface_audit.json').read_text())
    if index['status']!='complete' or checked['status']!='passed' or any(r['split']!='train' for r in index['records']):raise ValueError('Audited train dataset required')
    if digest(a.dataset/'inputs_observable.npz')!=index['inputs_sha256'] or digest(a.dataset/'labels_offline.json')!=index['labels_sha256']:raise ValueError('Dataset changed')
    run=output_path(a.output_dir) if a.output_dir else new_run('pulse_siglip',group=('feature_caches',));run.mkdir(parents=True,exist_ok=True);print(run,flush=True);setup_imports(run)
    with (run/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model,encoder=load_encoder(a.device)
        contract=dict(dataset=str(a.dataset.resolve()),dataset_index_sha256=digest(a.dataset/'index.json'),checkpoint=str(CHECKPOINT),encoder=encoder,device=a.device,limit=a.limit,code_sha256=digest(Path(__file__)),encoder_code_sha256=digest(Path(__file__).with_name('cache_siglip.py')),scope='Only actual common pulse-start and previous observed RGB. Pooled 3x4x4x1152, no language/GT/future image features.')
        if (run/'manifest.json').exists():
            if json.loads((run/'manifest.json').read_text())!=contract:raise ValueError('Cache contract changed')
        else:write_json(run/'manifest.json',contract)
        sources=index['sources'][:a.limit] if a.limit else index['sources'];records=[]
        for source_index,source in enumerate(sources):
            original=Path(source['run'])
            if digest(original/'result.json')!=source['result_sha256'] or digest(original/'manifest.json')!=source['manifest_sha256']:raise ValueError('Run changed')
            result=json.loads((original/'result.json').read_text());manifest=json.loads((original/'manifest.json').read_text());frames={f['offset']:f for f in result['branches']['base']['frames']}
            for role,offset in [('current',manifest['pulse_start']),('history',manifest['pulse_start']-4)]:
                src=Path(source[role+'_observed_file']);sha=digest(src);frame=frames[offset]
                if src!=original/frame['observed_file'] or sha!=frame['observed_sha256'] or sha!=source[role+'_observed_sha256']:raise ValueError('Wrong pulse/history frame')
                dest=run/f'source_{source_index:04d}_{role}.npz'
                if dest.exists():
                    with np.load(dest) as z:
                        if str(z['source_sha256'])!=sha or z['pooled'].shape!=(3,4,4,1152) or not np.isfinite(z['pooled']).all():raise ValueError('Invalid existing feature')
                else:
                    with np.load(src) as z:images=[z[k+'_rgb'] for k in ('head','right','left')]
                    _,pooled=encode(model,images,a.device)
                    tmp=dest.with_suffix('.tmp')
                    with tmp.open('wb') as f:np.savez_compressed(f,pooled=pooled,source_sha256=sha)
                    tmp.replace(dest)
                records.append(dict(source_index=source_index,role=role,offset=offset,file=str(dest),sha256=digest(dest),source=str(src),source_sha256=sha))
            replace_json(run/'progress.json',dict(completed_frames=len(records),total_frames=2*len(sources)));print(f'CACHED {source_index+1}/{len(sources)}',flush=True)
        replace_json(run/'result.json',dict(status='complete',limited=a.limit is not None,records=records));print(f'COMPLETE frames={len(records)} images={len(records)*3}',flush=True)

if __name__=='__main__':main()
