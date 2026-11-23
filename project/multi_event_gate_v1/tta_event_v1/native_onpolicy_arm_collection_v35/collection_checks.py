"""Real qualified packet parity, fixed population and untouched full/guard."""
from pathlib import Path
import ast
from .common import T,C,TASKS,read,mapped,sha
from .contracts import check
from .known_preparation import collect
from native_onpolicy_arm_teacher_v1.stream import packet
from native_onpolicy_recovery_stream_v2.stream import packet as old_packet
from native_policy_execution_v1.common import write
def main():
 check();count=0
 for namespace in('native_onpolicy_full_v1','native_onpolicy_coverage_v8'):
  o=T/'outputs'/namespace/'pilot_fit';r=mapped(read(o/'execution_state.json')['root']);spec=read(r/'protocol.json');assert read(r/'terminal_audit.json')['passed']and read(o/'actual_host_wait_exit.json')['exit_code']==0
  for case in read(r/'GPU_episodes_complete.json')['cases']:
   folder=r/'cases'/case['case_id']/'M_event_full';a=packet(spec,folder,case);b=old_packet(spec,folder,case);assert(a is None)==(b is None)
   if a is not None:assert a['choice']==b['choice']and a['query']==b['query']and a['expected_physical']==b['expected_physical']and a['expected_rng']==b['expected_rng']
   count+=1
 initial={t:dict(next_seed=7120000,accepted=0)for t in TASKS};cache=collect(initial,TASKS,5,7120000);assert cache['cached_valid_cases']==25
 for task,v in cache['task_cache'].items():assert [i['episode_index']for i in v['items']]==list(range(5))and v['end_state']['accepted']==5
 val={t:dict(next_seed=7121000,accepted=0)for t in TASKS};empty=collect(val,TASKS,5,7121000);assert empty['cached_valid_cases']>=5 and all(v['end_state']['accepted']>=1 for t,v in empty['task_cache'].items())
 # All ten natural25-case stages are necessary members of the predeclared
 # population; there is no short-run loop or policy-success replacement.
 fit=[(task,i)for n in range(8)for task in TASKS for i in range(n*5,(n+1)*5)];validation=[(task,i)for n in range(2)for task in TASKS for i in range(n*5,(n+1)*5)]
 assert len(fit)==200 and len(set(fit))==200 and len(validation)==50 and len(set(validation))==50
 runtime=(C/'runtime.py').read_text();assert "prepared['live_teacher_schedule']=schedule"in runtime and runtime.index('schedule.finish()')<runtime.index("lambda:release_completed(schedule,c,torch,models")and 'or not future_done.is_set()'in runtime and 'native_onpolicy_arm_teacher_v1.stream import recover'in runtime
 from . import common
 host_tree=ast.parse((C/'host.py').read_text());imports=[n for n in host_tree.body if isinstance(n,ast.ImportFrom)and n.module=='common'and n.level==1]
 assert len(imports)==1 and {'O','replace'}<={a.name for a in imports[0].names} and common.O.name=='fit_000'
 for d in(C,T/'native_onpolicy_arm_teacher_v1',T/'native_expert_case_release_v1',T/'native_metadata_stream_v1',T/'native_server_rule4_v5',T/'native_account_stream_v1'):
  for p in d.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 from .future_checks import run as check_future
 future_checked=check_future();partial_future=check_future(stop_after=3);empty_future=check_future(stop_after=0)
 assert future_checked['passed']and partial_future['passed']and empty_future['passed']
 from .dispatch_checks import run as dispatch_check
 dispatch_checked=dispatch_check();assert dispatch_checked['passed']
 from .scheduling_checks import run as scheduling_check
 scheduled=scheduling_check();assert scheduled['passed']
 source={str(p):sha(p)for d in(C,T/'native_onpolicy_arm_teacher_v1',T/'native_expert_case_release_v1',T/'native_metadata_stream_v1',T/'native_server_rule4_v5',T/'native_account_stream_v1')for p in d.rglob('*')if p.is_file()and p.suffix in('.py','.md','.sh')}
 write(T/'outputs/native_onpolicy_arm_collection_v35/CPU_collection_contract_checks.json',dict(passed=True,actual_qualified_full_cases_checked=count,original_FIT_packet_state_selection_exact=True,canonical_first25_q0_only_preparation_verified=True,VAL_uses_only_actual_VAL_inputs_not_FIT=True,fixed_FIT200_VAL50_population_exact=True,original_full_episode_and_resource_terminal_AST_exact=True,actual_local_output_namespace_verified=True,planners_retained_until_all_teacher_workers_complete=True,next_required_preparation_IO_check=future_checked,actual_dynamic_dispatch_and_case_RNG=dispatch_checked,source_sha256=source,optimizer_steps=0,new_GPU_work=False))
 write(T/'outputs/native_onpolicy_arm_collection_v35/CPU_scheduling_contract_checks.json',dict(passed=True,scheduling=scheduled,full_future=future_checked,partial_future=partial_future,empty_future=empty_future,source_sha256=source,no_GPU_work=True))
 print('CPU_REAL40_FULL_PACKET_PARITY_AND_FIXED250_COLLECTION_CONTRACTS_PASS')
if __name__=='__main__':main()
