import os,sys,subprocess,importlib.util,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];D=R/'deployment';sys.path.insert(0,str(D))
from evaluate import CANONICAL_PROJECT,add_env
spec=importlib.util.spec_from_file_location('raw241',D/'user_container.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
cohort=int(os.environ.get('CX_GATE_TRANSFER_COHORT','0'));root=R/f'multi_event_gate_v1/outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4/candidate_veto_all_events_v237_{cohort}/dev0'
script=Path(sys.argv[1]);script=script if script.is_absolute()else D/script;script=CANONICAL_PROJECT/script.relative_to(R)
cmd=c.command(['bash',str(CANONICAL_PROJECT/'deployment/task_adapter_python.sh'),str(CANONICAL_PROJECT/HERE.relative_to(R)/'entry241.py'),str(script),*sys.argv[2:]],mount_project=True)
add_env(cmd,dict(CX_BENEFIT_FOLLOWUP_CONTAINER='1',CX_GATE_TRANSFER_COHORT=str(cohort),CX_BENEFIT_GROWTH_BLOCK='dev0',CX_ALL237_ORIGINAL_CPU='0',CX_TARGET_GPU_UUID='GPU-dabef67b-c231-f3b7-8b8a-e35035b6c31a',PYTORCH_NVML_BASED_CUDA_CHECK='1',TORCH_WARM_POOL='0',CX_TOPP218_CONFIG=os.environ.get('CX_TOPP218_CONFIG','')))
log=root/'migration241'/('CPU_'+str(time.time_ns())+'.log')
with log.open('w')as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
print('CPU241_EXIT',r.returncode,'LOG',log,flush=True);print(log.read_text()[-4000:],flush=True);raise SystemExit(r.returncode)
