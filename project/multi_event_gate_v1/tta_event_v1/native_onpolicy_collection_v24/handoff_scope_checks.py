"""Explicit source delta and actual CPU helper checks before native validation."""
from pathlib import Path
import ast
from .common import T,C,read,sha
from native_policy_execution_v1.common import write

def main():
 old=T/'native_onpolicy_collection_v14';changed={'AGENTS.md','runtime.py','closed_audit.py','collection_plan.py','known_preparation.py','prefix_stage.py','startup.py'}
 matched=[]
 for p in old.rglob('*'):
  if not p.is_file()or p.suffix not in('.py','.sh','.md'):continue
  r=p.relative_to(old);new=C/r
  if str(r)in changed:continue
  assert new.read_text()==p.read_text().replace('native_onpolicy_collection_v14','native_onpolicy_collection_v24'),str(r)
  matched.append(str(r))
 scene=read(T/'outputs/native_scene_free_topp_v2/CPU_actual_MPLIB_scene_release_checks.json');assert scene['passed']and scene['exact_original_TOPP_results']==20
 assert read(T/'outputs/native_scene_free_topp_v2/CPU_actual_host_exit.json')['exit_code']==0
 helper=read(T/'outputs/native_q0_task_handoff_v3/CPU_actual_checks.json');assert helper['passed']and helper['actual_CPU_PhysX_transfers']==10
 exit=read(T/'outputs/native_q0_task_handoff_v3/CPU_actual_host_exit_v2.json');assert exit['exit_code']==0 and exit['actual_Popen_wait_return']and exit['proc_absent_after_wait']and exit['CUDA_hidden']
 for p,h in helper['source_sha256'].items():
  from .common import mapped
  assert sha(mapped(p))==h
 for p in C.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 from .startup import StagedLanes
 import threading
 emitted=[];gate=StagedLanes(threading.Event(),emitted.append,lanes=10)
 for lane in range(4):assert gate.wait(lane)
 assert not gate.eight.is_set()and not gate.release('actual_once_policy_query_complete',lane=4,query=0,tick=0)
 assert gate.release('actual_once_policy_query_complete',lane=0,query=0,tick=0)and emitted[0]['released_lanes']==list(range(4,10))
 for lane in range(10):assert gate.wait(lane)
 from .contracts import check
 check()
 from .collection_checks import main as original_collection_checks
 original_collection_checks()
 write(T/'outputs/native_onpolicy_collection_v24/CPU_handoff_scope_checks.json',dict(passed=True,unchanged_source_files_after_namespace_only=matched,changed_files=sorted(changed),original_episode_and_sustained_resource_terminal_exact=True,actual_CPU_handoff_proof=helper,native_whole_reference_controls_required=True,full_250_and_minimum_support_gate_unchanged=True,no_GPU_or_optimizer=True,source_sha256={str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.md','.sh')}))
 print('Q0_HANDOFF_SCOPE_AND_REAL_OLD40_PACKET_FIXED250_CPU_PASS',flush=True)
if __name__=='__main__':main()
