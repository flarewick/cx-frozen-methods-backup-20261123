"""Run the frozen original entrypoints inside the RTX 4090 deployment image."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=Path('/data/cj/robo_pi/250010033')
PROJECT=BASE/'VLA-Adapter/cx_new_direction_guide_TTA'
RT=BASE/'qfw/robotwin'
W=PROJECT/'multi_event_gate_v1'


def command(args,forward,project=None):
    project=Path(project or HERE.parent).resolve();artifacts=args.artifacts.resolve()
    if forward[:1]==['--']:forward=forward[1:]
    if any(a=='--gpu' or a.startswith('--gpu=') or a=='--condition' or a.startswith('--condition=') for a in forward):
        raise ValueError('Select host GPU and condition through wrapper --gpu / --entry only')
    if args.entry in ('baseline','full','preflight'):
        condition='Z_zero' if args.entry=='baseline' else 'M_event_full'
        inner=['bash',str(W/'scripts/eval_event_full_worker.sh'),'--condition',condition,'--gpu','0',*forward]
        if args.entry=='preflight':inner.append('--dry-run')
    elif args.entry in ('tta','preflight-tta'):
        inner=['bash',str(W/'tta_event_v1/scripts/probe_tta.sh'),'--gpu','0',*forward]
        if args.entry=='preflight-tta':inner.append('--dry-run')
    elif args.entry in ('tta-direction','preflight-tta-direction'):
        inner=['bash',str(W/'tta_event_v1/direction_probe_v1/scripts/probe_direction.sh'),'--gpu','0',*forward]
        if args.entry=='preflight-tta-direction':inner.append('--dry-run')
    elif args.entry in ('adapter-train','preflight-adapter-train'):
        inner=['bash',str(W/'task_adapter_v1/scripts/task_adapter.sh'),'--gpu','0',*forward]
        if args.entry=='preflight-adapter-train':inner.append('--dry-run')
    elif args.entry in ('adapter-baseline','adapter-full'):
        condition='baseline' if args.entry=='adapter-baseline' else 'M_event_full'
        inner=['bash',str(W/'task_adapter_v1/scripts/eval.sh'),'--condition',condition,'--gpu','0',*forward]
    else:inner=['python','/opt/cx_deployment/doctor.py']+(['--cpu'] if args.entry=='doctor-cpu' else [])
    cpu=args.entry in ('preflight','preflight-tta','doctor-cpu','preflight-tta-direction','preflight-adapter-train')
    cpu=cpu or '--dry-run' in forward or '--help' in forward
    cmd=['docker','run','--rm','--init','--shm-size=8g','--user',f'{os.getuid()}:{os.getgid()}',
         '--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONNOUSERSITE=1',
         '--env','NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics']
    if sys.stdin.isatty() and sys.stdout.isatty():cmd+=['-it']
    if not cpu:cmd+=['--gpus','device='+str(args.gpu)]
    else:cmd+=['--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=']
    mounts=[(project,PROJECT,False)]
    for rel in ('RoboTwin/assets','checkpoints/pi05_motus_robotwin2/motus_robotwin2/1000000'):
        mounts.append((artifacts/'rootfs/qfw/robotwin'/rel,RT/rel,True))
    for host,target,readonly in mounts:
        if ',' in str(host):raise ValueError('Docker mount path must not contain a comma')
        if not host.is_dir():raise FileNotFoundError('Missing deployment directory: '+str(host))
        cmd+=['--mount',f'type=bind,src={host},dst={target}'+(',readonly' if readonly else '')]
    cmd += ['--workdir',str(PROJECT),args.image,*inner]
    return cmd


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--entry',choices=('doctor','doctor-cpu','preflight','preflight-tta','baseline','full','tta',
        'tta-direction','preflight-tta-direction','adapter-train','preflight-adapter-train',
        'adapter-baseline','adapter-full'),required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--image',default='cx-direction-tta:20261022-4090')
    p.add_argument('--print-command',action='store_true')
    p.add_argument('forward',nargs=argparse.REMAINDER,help='Original entrypoint arguments after --')
    a=p.parse_args()
    if a.gpu<0:p.error('GPU must be nonnegative')
    try:cmd=command(a,a.forward)
    except (ValueError,FileNotFoundError) as e:p.error(str(e))
    print(shlex.join(cmd),flush=True)
    if not a.print_command:raise SystemExit(subprocess.call(cmd))


if __name__=='__main__':main()
