"""Lifecycle/poison/thread contracts on actual saved command arrays, CPU only."""
from pathlib import Path
import hashlib,json,threading
import numpy as np,torch
from .bridge import ReadoutBridge
T=Path(__file__).resolve().parents[1];C=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not torch.cuda.is_initialized();report=json.loads((T/'outputs/native_expert_full_failure_v5/report.json').read_text());record=next(r for r in report['rows']if r['index']==30)
 capture=json.loads(Path(record['capture_file']).read_text());chunk=capture['chunks'][0];assert sha(chunk['file'])==chunk['sha256']
 with np.load(chunk['file'],allow_pickle=False)as z:row={k:z[k][0].copy()for k in('arm_position_target','arm_velocity_target','gripper_position_target','gripper_velocity_target','gripper_logical_command','gripper_joint_counts')}
 def valid(_):return{k:v.copy()for k,v in row.items()}
 def rejects(fn):
  try:fn()
  except (ValueError,RuntimeError):return True
  return False
 bridge=ReadoutBridge(valid,valid);assert rejects(lambda:bridge.capture(None));bridge.begin_case(record['case_id'],0)
 assert rejects(lambda:bridge.begin_case('nested',0));value=bridge.capture(None);assert all(np.array_equal(value[k],v)for k,v in row.items())
 child=[]
 def worker():
  assert rejects(lambda:bridge.capture(None));bridge.begin_case('CPU_contract_thread_only',1);bridge.capture(None);child.append(bridge.finish_case())
 thread=threading.Thread(target=worker);thread.start();thread.join(2);assert not thread.is_alive()and len(child)==1 and child[0]['native_calls']==1
 main_case=bridge.finish_case();assert main_case['case_id']==record['case_id']and main_case['native_calls']==main_case['paired_calls']==1
 assert rejects(bridge.finish_case)
 def wrong_value(_):
  r=valid(None);r['arm_velocity_target'][0]+=1.;return r
 bad=ReadoutBridge(valid,wrong_value);bad.begin_case('CPU_negative_value',0);assert rejects(lambda:bad.capture(None))and rejects(bad.finish_case)and rejects(lambda:bad.capture(None))
 def wrong_dtype(_):
  r=valid(None);r['gripper_logical_command']=r['gripper_logical_command'].astype(np.float32);return r
 bad=ReadoutBridge(valid,wrong_dtype);bad.begin_case('CPU_negative_dtype',0);assert rejects(lambda:bad.capture(None))and rejects(bad.finish_case)
 def failure(_):raise RuntimeError('CPU fixture reader failure')
 bad=ReadoutBridge(failure,valid);bad.begin_case('CPU_negative_original_exception',0);assert rejects(lambda:bad.capture(None))and rejects(bad.finish_case)
 assert not torch.cuda.is_initialized()
 out=T/'outputs/native_telemetry_bridge_v1/CPU_contract_checks.json';out.parent.mkdir(parents=True,exist_ok=False)
 result=dict(passed=True,CPU_only=True,actual_saved_command_array_fixture=chunk['file'],fixture_sha256=chunk['sha256'],
 lifecycle_reset=True,thread_case_isolation=True,value_dtype_mismatch_and_original_failure_poison_case=True,
 original_candidate_readers_mocked_for_lifecycle_only=True,real_native_GPU_bridge_NOT_claimed=True,approved_for_collector=False,
 source_sha256={str(p):sha(p)for p in C.glob('*.py')})
 with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print('READOUT_BRIDGE_LIFECYCLE_POISON_AND_THREAD_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
