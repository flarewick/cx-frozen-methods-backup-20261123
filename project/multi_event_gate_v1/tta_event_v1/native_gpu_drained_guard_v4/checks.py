"""No live-resource waiver: exact phase-limited AST inverse and negative controls."""
import ast,copy,json
from pathlib import Path
from .guard import drained,transformed,base_transformed

def main():
 T=Path(__file__).resolve().parents[1];C=Path(__file__).parent
 original=(T/'native_process_sampler_v13/strict_guard.py').read_text();new=transformed(original)
 counts=dict(assign=0,priority=0,memory=0,telemetry=0)
 class Undo(ast.NodeTransformer):
  def visit_Assign(self,n):
   if len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='post_GPU_drained':counts['assign']+=1;return None
   if len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='telemetry':
    keep=[(k,v)for k,v in zip(n.value.keys,n.value.values)if not(isinstance(k,ast.Constant)and k.value=='GPU_drained_CPU_tail')];assert len(keep)==len(n.value.keys)-1;n.value.keys=[k for k,v in keep];n.value.values=[v for k,v in keep];counts['telemetry']+=1
   return self.generic_visit(n)
  def visit_If(self,n):
   if ast.unparse(n.test)=="priority['stop_for_prior_or_unverified_occupant'] and (not post_GPU_drained)":n.test=n.test.values[0];counts['priority']+=1
   if ast.unparse(n.test)=='memory > max_memory_mib and (not post_GPU_drained)':n.test=n.test.values[0];counts['memory']+=1
   return self.generic_visit(n)
 restored=Undo().visit(copy.deepcopy(new));assert counts==dict(assign=1,priority=1,memory=1,telemetry=1),counts
 assert ast.dump(restored)==ast.dump(base_transformed(original))
 note=dict(accepted=True,exit_code=0,normal_child_reaped=True,proc_absent_rechecked=True,parent_alive_same_identity=True,retired_host_pid=19)
 assert drained([],{19},{19},note)
 assert not drained([19],{19},{19},note)and not drained([20],{19},{19},note)
 assert not drained([],set(),set(),note)and not drained([],{19,20},{19},note)and not drained([],{19},set(),note)
 for key in('accepted','normal_child_reaped','proc_absent_rechecked','parent_alive_same_identity'):
  bad=dict(note);bad[key]=False;assert not drained([],{19},{19},bad)
 for key,value in[('exit_code',1),('retired_host_pid',20)]:
  bad=dict(note);bad[key]=value;assert not drained([],{19},{19},bad)
 from native_governor180_v5 import checks as policy
 assert len(policy.checks)==39
 assert (C/'guard.py').read_text().replace('native_governor180_v5','native_governor180_v4')==(T/'native_gpu_drained_guard_v2/guard.py').read_text()
 proof=json.loads((T/'outputs/native_rule8_startup_v1/CPU_checks_v1.json').read_text());assert proof['passed']and proof['exact_policy_inverse']
 actual=json.loads((T/'outputs/native_rule8_startup_v1/CPU_actual_exit_v1.json').read_text());assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 r=T/'outputs/native_seven_gated_dev_v3/dev_gated/batches/000/runs/048c0fb7a04d4321bfeb826df7925312';rows=[json.loads(s)for s in(r/'attempt/resources.jsonl').read_text().splitlines()];last=[x for x in rows if'seconds'in x][-1]
 assert last['raw_foreign_pids']==[1033427]and last['gpu_arrival_order']['stop_for_prior_or_unverified_occupant']
 assert drained(last['gpu_pids'],{910680},set(last['normal_child_retired_pids']),last['normal_child_exit_proof'])
 out=T/'outputs'/C.name;out.mkdir(exist_ok=True)
 import hashlib
 f=out/'CPU_checks_v1.json';assert not f.exists();f.write_text(json.dumps(dict(passed=True,original_guard_AST_inverse_exact=True,live_unknown_failed_new_GPU_and_missing_proofs_rejected=True,actual_late_foreign_CPU_tail_reproduced=True,no_foreign_load_credited=True,policy_checks=policy.checks,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in C.glob('*')if p.is_file()}),indent=2)+'\n');print('VERIFIED_GPU_DRAIN_CPU_TAIL_GUARD_PASS')
if __name__=='__main__':main()
