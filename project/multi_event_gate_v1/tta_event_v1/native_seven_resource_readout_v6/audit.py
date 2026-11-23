"""Final qualified resources; keep failed time and never average per-sample maxima."""
import json
from pathlib import Path
from native_seven_official_v9.common import T,read,verify,sha
from native_seven_official_v9.qualification import qualified_stage
from native_seven_official_v9.adoption import folder as stage_folder,IMPORTED_PLAIN_SESSIONS,failed_second,failed_current,failed_coordination
from native_eval_throughput_acceptance_v2.audit import metrics

def aggregate(rows,key):
 values=[r['metrics'][key]for r in rows];n=sum(v['samples']for v in values);assert n>0
 weighted=lambda k:sum(v[k]*v['samples']for v in values)/n
 sm=weighted('SM_percent');mc=weighted('memory_controller_percent')
 return dict(samples=n,mean_SM_percent=sm,mean_memory_controller_percent=mc,MAX_of_separate_means_percent=max(sm,mc),mean_CPU_percent=weighted('CPU_percent'),mean_memory_MiB=weighted('memory_MiB'),missing_metrics_counted_zero=sum(v['missing_metrics_count']for v in values))

def main():
 final=T/'outputs/native_seven_campaign_v8/all_seven_official100_complete.json';v=read(final);verify(v)
 assert v['complete']and v['TTA_episodes']==v['paired_control_episodes']==700
 rows=[];inputs={str(final):sha(final)};failed=0.;whole=0
 for group in('plain','gated'):
  for i in range(17):
   folder=stage_folder(group,i);root,spec=qualified_stage(folder)
   release_file=root/'attempt/resources.jsonl';release=json.loads(release_file.read_text().splitlines()[-1]);assert release['own_gpu_released']and not release['remaining_own_gpu_pids']
   if i==0:failed+=spec['inherited_stage_GPU_seconds']
   failed+=spec.get('retained_failed_C5_second_GPU_seconds',0.)+spec.get('retained_failed_C6_second_GPU_seconds',0.)+spec.get('retained_failed_C7_second_GPU_seconds',0.)
   data=read(root/'CPU_episode_audit.json');whole+=data['actual_native_episodes']+data['closed_expert_task_lifetimes']['whole_native_reference_controls']
   rows.append(dict(group=group,stage=spec['stage'],actual_GPU_uuid=spec['required_gpu_uuid'],qualified_session_seconds=release['cleanup_seconds_after_start'],metrics=metrics(root)))
   for p in(folder/'actual_host_wait_exit.json',root/'protocol.json',root/'terminal_audit.json',root/'CPU_episode_audit.json',root/'CPU_governor180_resource_audit.json',root/'attempt/governor180.jsonl',root/'attempt/governor180_raw.jsonl',release_file):inputs[str(p)]=sha(p)
 expected_failed=6962.875+((failed_second()[2]['cleanup_seconds_after_start']+failed_current()[2]['cleanup_seconds_after_start']+failed_coordination()[2]['cleanup_seconds_after_start'])if IMPORTED_PLAIN_SESSIONS==1 else 0.)
 assert len(rows)==34 and abs(failed-expected_failed)<1e-8
 seconds=sum(r['qualified_session_seconds']for r in rows);assert seconds>0
 value=dict(passed=True,all34_sessions_actual_normal0_native_and_server180_qualified=True,only_actual_owned_GPU_process_samples=True,no_other_user_or_CPU_tail_load_credited=True,full_owned_lifetime=aggregate(rows,'full_owned_lifetime_raw'),all_post_creation300s=aggregate(rows,'all_post_creation300s_raw'),each_session_target80_met=all(r['metrics']['target80_achieved']for r in rows),qualified_session_seconds=seconds,retained_failed_formal_session_seconds=failed,all_formal_session_seconds=seconds+failed,qualified_primary_episodes=1400,qualified_whole_episodes_including_controls=whole,primary_episodes_per_all_formal_session_hour=1400*3600/(seconds+failed),whole_episodes_per_qualified_session_hour=whole*3600/seconds,no_matched_speedup_claim=True,no_TTA_efficacy_claim_from_resources=True,rows=rows,input_sha256=inputs)
 value['by_group']={group:dict(full_owned_lifetime=aggregate([r for r in rows if r['group']==group],'full_owned_lifetime_raw'),all_post_creation300s=aggregate([r for r in rows if r['group']==group],'all_post_creation300s_raw'))for group in('plain','gated')}
 value['input_sha256'][str(Path(__file__))]=sha(Path(__file__))
 out=T/'outputs/native_seven_resource_readout_v6';out.mkdir(exist_ok=False);(out/'resources.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in value.items()if k not in('rows','input_sha256')},ensure_ascii=False))
if __name__=='__main__':main()
