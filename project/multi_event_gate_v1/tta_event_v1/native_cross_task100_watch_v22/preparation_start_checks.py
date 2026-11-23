"""Exercise full recovery orchestration with mocked systemctl/inventory; no GPU."""
from pathlib import Path
import json,types,copy
from unittest.mock import patch
from . import preparation_recovery as recovery
from .controller import T,O,read,save
from native_cross_task100_v19.preparation import collect
def main():
 old=T/'outputs/native_cross_task100_v11/batches/004/runs/94f5e547661349e4adcdffe374413880';spec=read(old/'protocol.json');study=read(T/'outputs/native_cross_task100_v19/study.json');prepared=collect(study,4,spec['initial_screen_state']);assert prepared['cached_valid_cases']==8
 original_exists=Path.exists
 root=O/'CPU_start_orchestration_fixture';previous=root/'batches/004/runs/synthetic';previous.mkdir(parents=True);save(previous/'protocol.json',spec);(previous/'attempt').mkdir();(previous/'attempt/resources.jsonl').write_text((old/'attempt/resources.jsonl').read_text());(previous/'GPU_host.log').write_text('')
 state=dict(ActiveState='failed',MainPID='0',InvocationID='CPU_fixture_only');control=dict(recoveries={},handled_invocations=[]);events=[]
 def event(kind,**data):events.append(dict(time='CPU_fixture',kind=kind,**data))
 def anchored_read(path):return dict(run_id=previous.name,traceback='RuntimeError: Per-process cumulative/rolling GPU utilization is unverified or below 34%')if Path(path).name=='campaign_failure.json'else read(path)
 with patch.object(Path,'exists',lambda self:True if self.name=='campaign_failure.json'else original_exists(self)),patch('native_cross_task100_v19.preparation.collect',return_value=prepared),patch('native_fast_inventory_v1.inventory.inventory',return_value=dict(own_gpu_count=0,available_indices=[5])),patch.object(recovery.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='',stderr=''))as start:
  okay=recovery.run(previous,state,control,T=T,O=root,UNIT='cx-tta-cross-task100-v19.service',ENV={},read=anchored_read,save=save,event=event,unit=lambda _:state,live_identity=lambda _:None)
  assert okay and start.call_count==1 and start.call_args.args[0]==['systemctl','--user','start','--no-block','cx-tta-cross-task100-v19.service']
 assert control['preparation_recoveries']=={'004':1}and len(events)==1
 proof=read(root/'verified_preparation_recovery_004_1.json');assert proof['old_completed_expert_rows']==0 and proof['new_completed_expert_rows']>=13 and proof['whole_primary_branches_must_rerun']and proof['resource_thresholds_and_GPU_clock_unchanged']
 # A changed invocation/live service must never receive a launch request.
 with patch.object(Path,'exists',lambda self:True if self.name=='campaign_failure.json'else original_exists(self)),patch('native_cross_task100_v19.preparation.collect',return_value=prepared),patch('native_fast_inventory_v1.inventory.inventory',return_value=dict(own_gpu_count=0,available_indices=[5])),patch.object(recovery.subprocess,'run')as start:
  assert not recovery.run(previous,state,dict(recoveries={},handled_invocations=[]),T=T,O=root,UNIT='cx-tta-cross-task100-v19.service',ENV={},read=anchored_read,save=save,event=event,unit=lambda _:dict(ActiveState='active',InvocationID='another'),live_identity=lambda _:None)
  start.assert_not_called()
 # A later CPU-only failure must never reuse an older GPU failure as its cause.
 with patch.object(Path,'exists',lambda self:True if self.name=='campaign_failure.json'else original_exists(self)),patch.object(recovery.subprocess,'run')as start:
  def wrong_run(path):return dict(run_id='different_failed_host',traceback='RuntimeError: Per-process cumulative/rolling GPU utilization is unverified or below 34%')if Path(path).name=='campaign_failure.json'else read(path)
  assert not recovery.run(previous,state,dict(recoveries={},handled_invocations=[]),T=T,O=root,UNIT='cx-tta-cross-task100-v19.service',ENV={},read=wrong_run,save=save,event=event,unit=lambda _:state,live_identity=lambda _:None)
  start.assert_not_called()
 with patch.object(Path,'exists',lambda self:True if self.name=='campaign_failure.json'else original_exists(self)),patch.object(recovery.subprocess,'run')as start:
  def unknown_cpu_failure(path):return dict(run_id=previous.name,traceback='RuntimeError: Unknown CPU preparation failure')if Path(path).name=='campaign_failure.json'else read(path)
  assert not recovery.run(previous,state,dict(recoveries={},handled_invocations=[]),T=T,O=root,UNIT='cx-tta-cross-task100-v19.service',ENV={},read=unknown_cpu_failure,save=save,event=event,unit=lambda _:state,live_identity=lambda _:None)
  start.assert_not_called()
 save(O/'CPU_preparation_start_checks.json',dict(passed=True,actual_CPU_verified_preparation_used=True,mocked_systemctl_start_once=True,recovery_counter_and_event_record_written=True,live_or_changed_invocation_never_launched=True,failed_host_run_must_match_previous_GPU_attempt=True,known_resource_RuntimeError_allowed_but_unknown_CPU_error_rejected=True,all_primary_branches_rerun_required=True,no_actual_service_calls=True,no_GPU_calls=True))
 print('STRICT_PREPARATION_RECOVERY_ORCHESTRATION_CPU_PASS',flush=True)
if __name__=='__main__':main()
