"""Bounded local S2g geometry/stage readout probe; no action policy training."""
import argparse,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json,replace_json
from .probe_candidates import read,verify
from .audit import digest
from .structured_readout_probe import StructuredProbe,load_probes,probe_loss,prepare_probes
from .train_rotation_curriculum import state_hash


def train(run,device):
    p,q,raw,labels=load_probes(run)
    assert read(run/'probe_preflight.json')['passed']
    if (run/'probe_environment.json').exists():raise ValueError('Fresh probe run required')
    x={k:torch.from_numpy(v).to(device) for k,v in raw.items()};reg=torch.from_numpy(labels['regression']).to(device);phase=torch.from_numpy(labels['phase']).to(device)
    write_json(run/'probe_environment.json',dict(torch=torch.__version__,device=device,threads=torch.get_num_threads(),
        gpu=None if device=='cpu' else torch.cuda.get_device_name(0),deterministic=torch.are_deterministic_algorithms_enabled(),
        parameters=q['parameters'],policy_backbone_loaded=False))
    started=time.monotonic();reports=[];signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Readout probe exceeded600seconds')));signal.alarm(600)
    try:
        for job in q['jobs']:
            train_ids=torch.tensor(job['train'],device=device);weights=torch.tensor(job['train_weights'],dtype=torch.float32,device=device)
            batch={k:v[train_ids] for k,v in x.items()}
            for seed in q['seeds']:
                initial=next(i for i in q['initializations'] if (i['scene'],i['seed'])==(job['scene'],seed))
                for mode in q['modes']:
                    model=StructuredProbe(p['channel_dim'],job['mean'],job['phase_prior']).to(device)
                    model.load_state_dict(torch.load(initial['path'],map_location=device,weights_only=True));assert state_hash(model)==initial['state_sha256']
                    opt=torch.optim.AdamW(model.parameters(),lr=q['lr'],weight_decay=q['weight_decay']);history=[]
                    tag=f'{job["scene"].split(":")[-1]}_{mode}_s{seed}'
                    for update in range(1,q['updates']+1):
                        opt.zero_grad();out=model(batch,mode);loss=probe_loss(out,reg[train_ids],phase[train_ids],weights)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
                        loss.backward();opt.step()
                        if update==1 or update%200==0:
                            history.append(dict(update=update,loss=float(loss.detach())))
                            replace_json(run/'probe_progress.json',dict(tag=tag,update=update,completed_models=len(reports),elapsed_seconds=time.monotonic()-started))
                    model.eval();checkpoint=run/(tag+'.pt');torch.save(model.state_dict(),checkpoint)
                    with torch.no_grad():output={k:v.cpu().numpy() for k,v in model(x,mode).items()}
                    artifact=run/(tag+'.npz');np.savez_compressed(artifact,**output)
                    reports.append(dict(scene=job['scene'],seed=seed,mode=mode,initial_sha256=initial['state_sha256'],
                        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),artifact=str(artifact),sha256=digest(artifact),
                        updates=q['updates'],history=history,final_loss=float(loss.detach())))
                    print('PROBE',tag,'completed',len(reports),'loss',float(loss.detach()),flush=True)
        verify(q);verify(p)
        write_json(run/'probe_result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-started,
            peak_gpu_memory_bytes=None if device=='cpu' else torch.cuda.max_memory_allocated(),
            scope='Diagnostic readout training only, no action adapter/LoRA update, simulator, router or TTA'))
        replace_json(run/'probe_progress.json',dict(phase='complete',models=len(reports)));print('PROBE_TRAINING_COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'probe_failure.json',dict(traceback=traceback.format_exc(),completed_models=len(reports)));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);parser.add_argument('--prepare',action='store_true');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    (prepare_probes(args.run.resolve()) if args.prepare else train(args.run.resolve(),args.device))
