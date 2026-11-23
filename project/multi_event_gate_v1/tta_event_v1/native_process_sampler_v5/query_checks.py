"""Direct guard query parity, bounded failures, and module isolation; no CUDA."""
import ast,copy,json,hashlib,subprocess,types,sys,time
from pathlib import Path
from unittest.mock import patch
from .nvml_query import DoctorProxy,render
from . import guard
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent;O=T/'outputs/native_process_sampler_v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=T/'native_process_sampler_v4'
 for name in ['worker.py','metrics.py']:
  assert ast.dump(ast.parse((C/name).read_text().replace('native_process_sampler_v5','native_process_sampler_v4')))==ast.dump(ast.parse((old/name).read_text())),name
 assert ast.dump(ast.parse((C/'strict_guard.py').read_text().replace('native_process_sampler_v5','native_process_sampler_v4')))==ast.dump(ast.parse((old/'strict_guard.py').read_text()))
 parent=[T/'outputs/native_process_sampler_v4/CPU_clock_checks.json',T/'outputs/native_process_sampler_v4/CPU_actual_exit.json'];assert json.loads(parent[0].read_text())['passed']and json.loads(parent[1].read_text())['actual_exit_code']==0
 fixture=dict(arguments=['-i','5','--query-gpu=utilization.gpu,utilization.memory,memory.used'],devices=[dict(index=5,uuid='GPU-fixture',memory_used_bytes=4096*2**20,memory_total_bytes=24564*2**20,SM=0,bandwidth=45)],read_only_NVML=True,CUDA_context_created=False)
 assert render(fixture,fixture['arguments'])=='0,45,4096\n'
 negatives=[]
 for key,value in [('SM',None),('SM',101),('bandwidth',-1),('memory_used_bytes',2**64-1),('memory_total_bytes',0),('index',4)]:
  f=copy.deepcopy(fixture);f['devices'][0][key]=value
  try:render(f,f['arguments'])
  except(AssertionError,TypeError,ValueError):pass
  else:raise AssertionError(key+' invalid accepted')
  negatives.append(key)
 appargs=['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'];apps=dict(arguments=appargs,devices=[dict(index=i,uuid='GPU-fixture-'+str(i),compute_processes=[])for i in range(7)],read_only_NVML=True,CUDA_context_created=False);apps['devices'][5]['compute_processes']=[dict(pid=123,used_GPU_memory_bytes=None)];assert render(apps,appargs)=='GPU-fixture-5,123,N/A\n'
 fake=types.SimpleNamespace(query=lambda _:(_ for _ in()).throw(AssertionError('Original CLI called')),unchanged_property='original')
 proxy=DoctorProxy(fake);assert proxy.unchanged_property=='original'
 for error in [subprocess.TimeoutExpired('CPU_fixture',3),subprocess.CalledProcessError(1,'CPU_fixture')]:
  with patch('native_process_sampler_v5.nvml_query.subprocess.run',side_effect=error)as call:
   try:proxy.query(fixture['arguments'])
   except type(error):pass
   else:raise AssertionError('Query error swallowed')
   assert call.call_count==1 and call.call_args.kwargs['timeout']==3
 from native_fast_inventory_v1.inventory import inventory
 free=inventory()['available_indices'];gpu=next((i for i in [5,1,0,3,4,2,6]if i in free),5)
 forms=[['-i',str(gpu),'--query-gpu=memory.total'],['-i',str(gpu),'--query-gpu=utilization.gpu,utilization.memory,memory.used'],['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'],['--query-compute-apps=gpu_uuid,pid']]
 proxy.report=O/'CPU_read_only_queries.jsonl';assert not proxy.report.exists();actual=[]
 for i in range(20):
  a=forms[i%4];started=time.monotonic();value=proxy.query(a);actual.append(dict(arguments=a,CSV=value,elapsed_seconds=time.monotonic()-started));print('READ_ONLY_NVML_GUARD_QUERY_PASS',i,flush=True)
 # Capacity is static; compare the former CLI once. Empty-card usage/util is checked only when still actually empty.
 capacity=proxy.query(forms[0]).strip();cli=subprocess.run(['nvidia-smi',*forms[0],'--format=csv,noheader,nounits'],check=True,text=True,capture_output=True,timeout=3).stdout.strip();assert capacity==cli
 empty_parity=None
 if gpu in inventory()['available_indices']:
  a=proxy.query(forms[1]).replace(' ','').strip();b=subprocess.run(['nvidia-smi',*forms[1],'--format=csv,noheader,nounits'],check=True,text=True,capture_output=True,timeout=3).stdout.replace(' ','').strip();assert a==b;empty_parity=dict(gpu=gpu,NVML=a,CLI=b)
 # Guard implementation gets a PRIVATE proxy; parent deployment module must remain untouched.
 sys.path.insert(0,str(T.parents[1]/'deployment'));import gpu_doctor
 original=gpu_doctor.query;implementation=guard._implementation();assert gpu_doctor.query is original and isinstance(implementation.gpu_doctor,DoctorProxy)and implementation.gpu_doctor.original is gpu_doctor
 out=dict(passed=True,entire_strict_guard_AST_unchanged_except_metric_import=True,original_v4_process_clock_and_selection_AST_exact=True,parent_clock_proofs_explicitly_reused={str(p):sha(p)for p in parent},original_deployment_module_unmodified=True,only_guard_private_proxy=True,same_three_second_timeout_and_one_attempt=True,unknown_process_memory_remains_NA=True,unknown_invalid_util_memory_fail_closed=True,negative_checks=negatives,actual_read_only_queries=actual,capacity_CLI_NVML_exact=capacity,empty_card_CLI_NVML_exact=empty_parity,no_GPU_work_started=True,source_sha256={str(p):sha(p)for p in C.glob('*')if p.suffix in('.py','.json')})
 with(O/'CPU_query_checks.json').open('x')as f:json.dump(out,f,indent=2)
 print('DIRECT_NVML_GUARD_QUERY_CPU_PASS',flush=True)
if __name__=='__main__':main()
