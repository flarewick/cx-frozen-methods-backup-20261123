"""Strictly paired full-batch recovery training; no candidate or checkpoint selection."""
import argparse
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json,replace_json
from .audit import digest
from .probe_candidates import read,verify
from .matched_recovery_data import load,prepare
from .train_scene_crossfit import candidate_loss
from .time_encoding import TimeEncodedAdapter
from .train_rotation_curriculum import state_hash
from .pose_adapter import TorchFK
from .robot_kinematics import RobotFK


def train(run,device):
    p,arrays,contexts,schedule=load(run)
    if not read(run/'preflight_audit.json')['passed']:raise ValueError('Preflight required')
    if (run/'environment.json').exists():raise ValueError('New isolated run required')
    data={k:torch.from_numpy(v).to(device) for k,v in arrays.items()}
    context=torch.from_numpy(contexts).to(device);ids=torch.from_numpy(schedule['train_indices']).to(device)
    pairs=torch.from_numpy(schedule['pair_indices']).to(device)
    robots=[TorchFK(RobotFK(read(Path(p['config'])),s)).float().to(device) for s in ('left','right')]
    write_json(run/'environment.json',dict(device=device,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=None if device=='cpu' else torch.cuda.get_device_name(0),threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,full_batch_rows=len(ids)))
    reports=[];start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Matched training exceeded1800seconds')))
    signal.alarm(1800)
    try:
        for initial in p['initializations']:
            for ci,condition in enumerate(p['conditions']):
                encoding,seed=initial['encoding'],initial['seed'];tag=f'{condition}_{encoding}_s{seed}'
                torch.manual_seed(seed);model=TimeEncodedAdapter(157,encoding=encoding,envelope=True).to(device)
                model.load_state_dict(torch.load(initial['path'],map_location=device,weights_only=True))
                if state_hash(model)!=initial['state_sha256']:raise ValueError('Paired initialization changed')
                opt=torch.optim.AdamW(model.parameters(),lr=p['lr'])
                weights=torch.from_numpy(schedule['row_weights'][ci]).to(device)[ids]
                pw=torch.from_numpy(schedule['pair_weights'][ci]).to(device)
                history=[]
                for update in range(1,p['updates']+1):
                    opt.zero_grad();pred=model(context[ids],data['base'][ids])
                    loss=candidate_loss(pred,data['target'][ids],data['valid'][ids],weights,pairs,pw,robots,model.caps)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();opt.step()
                    if update==1 or update%200==0:
                        history.append(dict(update=update,loss=float(loss.detach())))
                        replace_json(run/'progress.json',dict(phase='training',tag=tag,update=update,
                            completed_models=len(reports),elapsed_seconds=time.monotonic()-start))
                model.eval();checkpoint=run/(tag+'.pt');torch.save(model.state_dict(),checkpoint)
                restored=TimeEncodedAdapter(157,encoding=encoding,envelope=True).to(device)
                restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));restored.eval()
                predicted=[]
                with torch.no_grad():
                    for offset in range(0,len(context),64):
                        x,b=context[offset:offset+64],data['base'][offset:offset+64]
                        values=model(x,b)
                        if not torch.equal(values,restored(x,b)):raise ValueError('Reload changed predictions')
                        predicted.append(values.cpu().numpy())
                artifact=run/(tag+'.npz');np.savez_compressed(artifact,predicted=np.concatenate(predicted),indices=np.arange(len(context)))
                write_json(run/(tag+'_history.json'),history)
                reports.append(dict(condition=condition,encoding=encoding,seed=seed,
                    initial_sha256=initial['state_sha256'],updates=p['updates'],final_loss=history[-1]['loss'],
                    checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),artifact=str(artifact),sha256=digest(artifact)))
                print('MODEL',tag,'loss',history[-1]['loss'],'completed',len(reports),flush=True)
        verify(p)
        write_json(run/'result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-start,
            peak_gpu_memory_bytes=None if device=='cpu' else torch.cuda.max_memory_allocated(),scope=p['scope']))
        replace_json(run/'progress.json',dict(phase='complete',models=len(reports)))
        print('TRAINING_COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_models=len(reports)))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run',type=Path);parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.prepare_only:prepare()
    elif args.run is None:parser.error('--run required')
    else:train(args.run.resolve(),args.device)
