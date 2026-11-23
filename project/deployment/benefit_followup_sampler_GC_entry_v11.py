"""Add explicit first-probe GC scheduling to the task-affine host entry."""
import os,sys,time,json,marshal,hashlib
from pathlib import Path
_INSTALLED=False

def install():
 global _INSTALLED
 if _INSTALLED:return
 import benefit_followup_affinity_entry_v10 as parent;parent.install()
 import benefit_followup_protocol_v2 as p
 proof=p.HOME/'sampler_GC_CPU_proof_v11.json';value=json.loads(proof.read_text())
 if not value['passed']or value['helper_sha256']!=p.sha(p.D/'benefit_sampler_gc_v11.py')or value['checker_sha256']!=p.sha(p.D/'verify_benefit_sampler_gc_cpu_v11.py'):raise RuntimeError('Sampler GC CPU proof stale')
 amendment=p.HOME/'sampler_GC_resource_amendment_v11.json'
 if not amendment.exists():raise RuntimeError('Explicit first-probe GC amendment required')
 __import__('benefit_sampler_gc_v11').install()
 info=dict(schema='first_probe_host_GC_coalescing_v11',entry_sha256=p.sha(__file__),helper_sha256=p.sha(p.D/'benefit_sampler_gc_v11.py'),CPU_proof_sha256=p.sha(proof),amendment_sha256=p.sha(amendment),original_ConstructorGC_sha256=p.sha(p.D/'collection_gc_schedule.py'),original83_runtime_files_unchanged=True,service_execute_wrapped_for_GC_only=True,original_model_callables_RNG_math_backend_untouched=True,first_original_probe_only=True,before_and_after_full_GC=True,original_non_graph_GC_retained=True,original_errors_propagate=True,all_resource_guards_unchanged=True)
 atomic=p.atomic
 def record(path,data):
  if Path(path).name=='deployment_cpu_schedule.json':data=dict(data,sampler_GC_schedule_v11=info)
  return atomic(path,data)
 p.atomic=record
 if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
  import user_container as c
  from evaluate import CANONICAL_PROJECT,host
  before=c.command;old=str(CANONICAL_PROJECT/'deployment/benefit_followup_affinity_entry_v10.py');new=str(CANONICAL_PROJECT/'deployment/benefit_followup_sampler_GC_entry_v11.py')
  def route(argv,*a,**kw):
   out=before(argv,*a,**kw)
   for i,x in enumerate(out):
    if x==old:out[i]=new
    elif isinstance(x,str)and "__import__('benefit_followup_affinity_entry_v10').install();"in x:out[i]=x.replace("__import__('benefit_followup_affinity_entry_v10').install();","__import__('benefit_followup_sampler_GC_entry_v11').install();",1)
   values=list(argv)
   if '--barrier'in values:
    folder=host(values[values.index('--barrier')+1]);folder.resolve().relative_to(p.ROOT.resolve());atomic(folder/'sampler_GC_host_schedule_v11.json',info)
   return out
  c.command=route
 else:
  from model_service import ModelService
  folder=p.ROOT/'sampler_GC_entry_receipts_v11';folder.mkdir(exist_ok=True)
  atomic(folder/('entry_'+str(time.time_ns())+'.json'),dict(info,pid=os.getpid(),created_unix=time.time(),actual_execute_wrapper_module=ModelService.execute.__module__,actual_execute_code_sha256=hashlib.sha256(marshal.dumps(ModelService.execute.__code__)).hexdigest()))
 _INSTALLED=True

def main():
 install();__import__('benefit_followup_affinity_entry_v10').main()
if __name__=='__main__':main()
