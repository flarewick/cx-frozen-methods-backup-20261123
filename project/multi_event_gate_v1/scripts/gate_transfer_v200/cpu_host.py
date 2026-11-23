"""Explicit no-device container runner. Logs only inside new transfer cohort."""
import os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import entry;entry.install()
import benefit_followup_protocol_v2 as p,user_container as c
from evaluate import CANONICAL_PROJECT,add_env
script=Path(sys.argv[1]);script=script if script.is_absolute()else p.D/script
if script.name=='benefit_event_full_scope_audit_v175.py':script=HERE/'audit_scope.py'
canonical=CANONICAL_PROJECT/script.relative_to(p.R)
cmd=c.command(['bash',str(CANONICAL_PROJECT/'deployment/task_adapter_python.sh'),str(canonical),*sys.argv[2:]],mount_project=True)
add_env(cmd,dict(CX_GATE_TRANSFER_COHORT=str(entry.COHORT),CX_TRANSFER_ORIGINAL_PLANNER_CHECK=os.environ.get('CX_TRANSFER_ORIGINAL_PLANNER_CHECK','0'),CX_TARGET_GPU_UUID='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a',CX_BENEFIT_GROWTH_BLOCK='dev0',PYTORCH_NVML_BASED_CUDA_CHECK='1',TORCH_WARM_POOL='0',JE_ARROW_MALLOC_CONF='background_thread:false',MALLOC_CONF='background_thread:false',ARROW_DEFAULT_MEMORY_POOL='system'))
log=p.ROOT/('CPU_'+script.stem+'_'+str(time.time_ns())+'.log')
with log.open('w')as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
print('CPU_EXIT',r.returncode,'LOG',log,flush=True);print(log.read_text()[-4500:],flush=True);raise SystemExit(r.returncode)
