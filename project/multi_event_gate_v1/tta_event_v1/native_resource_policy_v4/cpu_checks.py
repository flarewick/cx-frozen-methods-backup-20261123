"""Guard AST equality except fleet import, same quota/admission/failure contracts."""
from pathlib import Path
import ast,json,copy,hashlib
from .admission import require_admission
from . import guard
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent

def normalized(tree):
 class ImportOnly(ast.NodeTransformer):
  def visit_ImportFrom(self,node):
   if node.module=='native_fast_inventory_v1.inventory':return ast.copy_location(ast.ImportFrom(module='gpu_policy',names=[ast.alias(name='inventory'),ast.alias(name='MAX_GPUS')],level=0),node)
   if node.module=='gpu_policy'and[n.name for n in node.names]==['MAX_GPUS']:return None
   return node
 return ast.dump(ImportOnly().visit(tree),include_attributes=False)
def main():
 original=T/'native_terminal_guard_v1/strict_guard.py';assert normalized(ast.parse((C/'strict_guard.py').read_text()))==ast.dump(ast.parse(original.read_text()),include_attributes=False)
 assert guard.MIN_GPU_UTILIZATION==34 and guard.MAX_GPU_MEMORY_MIB==24563
 accepted=[];rejected=[]
 def fleet(count):return dict(max_gpus=3,own_gpu_count=count,cards=[dict(index=i,uuid='GPU-test-'+str(i),available=i>=count,pids=[100+i]if i<count else[],own_pids=[100+i]if i<count else[],sm=0,memory_mib=4096 if i<count else 1,total_mib=24564)for i in range(7)])
 for count in(0,1,2):assert require_admission(fleet(count),'GPU-test-5')['index']==5;accepted.append(count)
 for name,change in(('account3',lambda x:x.update(own_gpu_count=3,cards=fleet(3)['cards'])),('occupied',lambda x:x['cards'][5].update(pids=[999])),('memory65',lambda x:x['cards'][5].update(memory_mib=65)),('SM6',lambda x:x['cards'][5].update(sm=6)),('inconsistent',lambda x:x.update(own_gpu_count=2))):
  x=fleet(0);change(x)
  try:require_admission(x,'GPU-test-5')
  except RuntimeError:rejected.append(name)
  else:raise AssertionError(name)
 # Real implementation must load the hash-bound strict copy, same limits.
 implementation=guard._implementation();assert implementation.validate_memory_limit(24563,24564)==24563
 N=T/'native_selected_runtime_v2'
 for p in N.glob('*.py'):ast.parse(p.read_text())
 for name in('host.py','resources.py','launch.py'):
  s=(N/name).read_text();assert 'native_resource_policy_v4'in s and 'gpu_doctor.check_admission('not in s and 'from gpu_policy import inventory'not in s
 def execution(file):
  tree=ast.parse(file.read_text());return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute'),include_attributes=False)
 assert execution(N/'runtime.py')==execution(T/'native_selected_runtime_v1/runtime.py')
 report=dict(passed=True,strict_guard_AST_identical_except_inventory_import=True,native_execution_AST_identical_to_selected_v1=True,admission_accepts_own_counts=accepted,admission_rejected=rejected,all_future_admission_sites_v4=True,NVML_three_second_fail_closed_checks=str(T/'outputs/native_fast_inventory_v1/CPU_actual_inventory_checks.json'),GPU_work_started=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for folder in(C,N)for p in folder.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.json')})
 O=T/'outputs/native_resource_policy_v4';O.mkdir(exist_ok=False)
 with(O/'CPU_policy_and_guard_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('V4_SAME_GUARD_LIMITS_NVML_ONLY_PASS',accepted,rejected,flush=True)
if __name__=='__main__':main()
