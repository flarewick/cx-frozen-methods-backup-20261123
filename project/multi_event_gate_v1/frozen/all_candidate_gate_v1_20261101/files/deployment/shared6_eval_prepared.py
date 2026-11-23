"""One-thread CPU prelude; fresh CUDA child only after renewed host admission."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

def main():
    control=argparse.ArgumentParser(add_help=False);control.add_argument('--barrier',type=Path);control.add_argument('--cpu-proof',action='store_true');extra,remaining=control.parse_known_args()
    if not extra.cpu_proof and not extra.barrier:raise ValueError('GPU admission barrier required')
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','');os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    import shared6_eval_worker as worker
    from shared6_eval_backend import sources
    a=worker.arguments(remaining);prepared=worker.prepare_cpu(a)
    threads=len(list(Path('/proc/self/task').iterdir()))
    if threads!=1 or torch.cuda.is_initialized():raise RuntimeError('CPU prepared state is not one-thread/no-CUDA')
    ready=dict(task=a.task,gpu_uuid=a.gpu_uuid,os_threads=threads,cuda_initialized=False,source_sha256=sources(),plan=prepared['plan'],master_session=str(prepared['master'].run)if prepared['master']else None,gpu_needed=bool(prepared['lanes']))
    if extra.cpu_proof:
        child=os.fork()
        if child==0:
            assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()and int(torch.arange(10).sum())==45;os._exit(0)
        _,status=os.waitpid(child,0)
        if os.waitstatus_to_exitcode(status):raise RuntimeError('CPU fork-child validation failed')
        print('SHARED6_CPU_PROOF',json.dumps(dict(passed=True,cpu_only=True,cpu_fork_child_passed=True,**{k:ready[k]for k in('task','gpu_uuid','os_threads','cuda_initialized','source_sha256','gpu_needed')})),flush=True)
        if prepared['master']:prepared['master'].cleanup()
        return
    extra.barrier.mkdir(parents=True,exist_ok=True);(extra.barrier/'ready.json').write_text(json.dumps(ready)+'\n');began=time.monotonic()
    while not(extra.barrier/'permit.json').exists():
        if time.monotonic()-began>240:raise TimeoutError('No renewed host GPU admission permit')
        time.sleep(.05)
    permit=json.loads((extra.barrier/'permit.json').read_text())
    if permit['gpu_uuid']!=a.gpu_uuid:raise RuntimeError('GPU permit differs from original plan')
    child=os.fork()
    if child==0:
        try:
            os.environ['CUDA_VISIBLE_DEVICES']=selected
            if torch.cuda.is_initialized()or torch.cuda._is_in_bad_fork():raise RuntimeError('Unsafe CUDA fork')
            worker.run_gpu(a,prepared)
        except BaseException:traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
        os._exit(0)
    _,status=os.waitpid(child,0);raise SystemExit(os.waitstatus_to_exitcode(status))
if __name__=='__main__':main()
