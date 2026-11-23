"""Actual frozen VLA + SAM2-tiny CPU preload, before any CUDA fork.

The large DINO/SAM image teacher is absent from this GPU route. Earliest semantic
seeding remains a separate dependency; CPU preload is not co-residency proof.
"""
from pathlib import Path
import hashlib
import json
import os
import importlib.util
T=Path(__file__).resolve().parent.parent
O=T/'outputs/native_delayed_query_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_cpu():
    snapshots=[]
    def note(phase):
        rows=[dict(tid=int(p.name),name=(p/'comm').read_text().strip())for p in Path('/proc/self/task').iterdir()]
        snapshots.append(dict(phase=phase,threads=rows))
        print('CPU_MODEL_THREADS',phase,len(rows),rows,flush=True)
    note('before_torch')
    import torch
    note('after_torch')
    from adapter_cache_cpu import load_cpu_policy,state_digest
    from query_object_v1.stream import video
    note('after_video_import')
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise ValueError('Actual CPU-only preload required')
    before_env=dict(os.environ);before_affinity=set(os.sched_getaffinity(0))
    from native_delayed_query_v1.primitives import configuration,SOURCE_SHA256
    note('after_pure_primitives')
    assert before_env==dict(os.environ)and before_affinity==set(os.sched_getaffinity(0))
    base,info=load_cpu_policy(verify_state=True)
    note('after_actual_VLA_weights')
    before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):mask=video.build('cpu')
    note('after_actual_SAM2_weights')
    assert torch.equal(before,torch.random.get_rng_state())
    reference=json.loads((T/'outputs/native_arrived_mask_v3/cpu_models.json').read_text())
    mask_digest=state_digest(mask)
    assert mask_digest==reference['model_state_sha256']['SAM2']
    assert sum(p.numel()for p in mask.parameters())==38962498
    assert not mask.training and not any(p.requires_grad for p in mask.parameters())
    assert all(p.device.type=='cpu'for p in mask.state_dict().values())
    original=json.loads((T/'outputs/native_current_v4/cpu_preload_proof.json').read_text())
    assert original['passed']and not torch.cuda.is_initialized()
    actual_base=state_digest(base._model)
    log=T.parents[1]/'deployment/reports/target_adapter_cache_cpu_arrow_nvml_20261024.log'
    prior=next(json.loads(v.split(' ',1)[1])for v in log.read_text().splitlines()if v.startswith('ADAPTER_CACHE_CPU_PROOF '))
    assert actual_base==prior['converted_state_sha256']
    note('after_model_state_verification')
    if len(list(Path('/proc/self/task').iterdir()))!=1:
        O.mkdir(parents=True,exist_ok=True)
        p=O/'combined_preload_thread_failure.json'
        with p.open('x')as f:json.dump(dict(snapshots=snapshots,CUDA_initialized=torch.cuda.is_initialized(),fork_started=False,source_sha256=sha(__file__)),f,indent=2)
        raise ValueError('Combined actual preload not one OS thread; no fork or CUDA permitted')
    proof=dict(passed=True,actual_models=['frozen_VLA','SAM2.1_tiny'],SAM2_parameter_count=38962498,
       model_state_sha256=dict(VLA=actual_base,SAM2=mask_digest),CPU_RNG_preserved_for_SAM2=True,
       OS_threads=1,CUDA_initialized=False,CPU_affinity=sorted(os.sched_getaffinity(0)),thread_provenance=snapshots,
       import_preserves_environment_and_affinity=True,
       source_sha256={str(Path(__file__)):sha(__file__),**SOURCE_SHA256},
       input_sha256={str(p):sha(p)for p in(T/'outputs/native_arrived_mask_v3/cpu_models.json',log)},
       full_GPU_co_residency_proven=False,online_feedback_and_independent_confirmation=False,
       scope='Two real CPU model weights/import/one-thread/noCUDA only; no live actor/tracker or resource qualification.',**info)
    return dict(base=base,mask_model=mask,proof=proof)

def main():
    import torch
    loaded=load_cpu();child=os.fork()
    if child==0:
        try:
            assert not torch.cuda.is_initialized()and not torch.cuda._is_in_bad_fork()
            assert int(torch.arange(10).sum())==45
        except BaseException:os._exit(1)
        os._exit(0)
    _,status=os.waitpid(child,0);assert os.waitstatus_to_exitcode(status)==0
    proof=dict(loaded['proof'],actual_CPU_fork_wait0=True,
               script_sha256=sha(Path(__file__).parent/'scripts/cpu_models.sh'))
    O.mkdir(parents=True,exist_ok=True);p=O/'two_model_cpu_preload.json'
    text=json.dumps(proof,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    print('DELAYED_ACTUAL_VLA_SAM2_CPU_PRELOAD_FORK_PASS',flush=True)
if __name__=='__main__':main()
