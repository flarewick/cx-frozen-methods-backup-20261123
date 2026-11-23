"""Explicit no-device subprocess with project virtualenv and isolated237 scope."""
import os,sys,subprocess,importlib.util,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';sys.path.insert(0,str(D))
from evaluate import CANONICAL_PROJECT,add_env
spec=importlib.util.spec_from_file_location('raw_container237',D/'user_container.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
cohort=int(os.environ.get('CX_GATE_TRANSFER_COHORT','0'));root=R/f'multi_event_gate_v1/outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4/candidate_veto_all_events_v237_{cohort}/dev0'
args=[str(CANONICAL_PROJECT/HERE.relative_to(R)/'entry237.py'),*sys.argv[1:]]
for i,v in enumerate(args):
 if v.startswith(str(R)+'/'):args[i]=str(CANONICAL_PROJECT/Path(v).relative_to(R))
 if '/'not in v and v.endswith('.py'):args[i]=str(CANONICAL_PROJECT/'deployment'/v)
cmd=c.command(['bash',str(CANONICAL_PROJECT/'deployment/task_adapter_python.sh'),*args],mount_project=True)
add_env(cmd,dict(CX_BENEFIT_FOLLOWUP_CONTAINER='1',CX_GATE_TRANSFER_COHORT=str(cohort),CX_BENEFIT_GROWTH_BLOCK='dev0',CX_ALL237_ORIGINAL_CPU=os.environ.get('CX_ALL237_ORIGINAL_CPU','0'),CX_TRANSFER_ORIGINAL_PLANNER_CHECK=os.environ.get('CX_ALL237_ORIGINAL_CPU','0'),CX_TARGET_GPU_UUID='GPU-fa87bc41-1919-efe1-73f9-11f38008763e',PYTORCH_NVML_BASED_CUDA_CHECK='1',TORCH_WARM_POOL='0',CX_TOPP218_CONFIG=os.environ.get('CX_TOPP218_CONFIG','')))
log=root/('CPU237_'+str(time.time_ns())+'.log')
with log.open('w')as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
print('CPU237_EXIT',r.returncode,'LOG',log,flush=True);print(log.read_text()[-4500:],flush=True);raise SystemExit(r.returncode)
