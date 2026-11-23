"""New split GPU0/CPU0 certificate; original outer host1 remains unchanged."""
from pathlib import Path
from .common import O,ROOT,B,T,read,sha,verify,write

def main():
 for name in('actual_resource_host_exit_v1.json','actual_CPU_recovery_exit_v1.json'):
  a=read(O/name);assert a['exit_code']==0 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 r=read(O/'result.json');verify(r);assert r['passed']and r['exact_36_whole_branches_replayed']and r['original_host_exit_code']==1
 assert read(B/'actual_host_wait_exit.json')['exit_code']==1
 normal=read(ROOT/'attempt/barrier/normal_child_exit.json');assert normal['normal_exit']and normal['exit_code']==0 and normal['proc_absent_after_wait']
 spec=read(ROOT/'protocol.json');verify(spec)
 audit=read(O/'audits/CPU_episode_audit.json');assert audit['passed']and len(audit['records'])==8 and audit['original_zero_shadow_controls_exact']
 groups={}
 for task in spec['tasks']:
  rows=[v for v in audit['records']if v['task']==task];assert len(rows)==2
  groups[task]=dict(scenes=2,full=sum(v['full']for v in rows),TTA=sum(v['TTA']for v in rows),rescues=sum(v['rescue']for v in rows),harms=sum(v['harm']for v in rows),accepted_update_scenes=sum(v['branches']['TTA']['accepted_updates']>0 for v in rows),changed_action_scenes=sum(v['first_changed_query']is not None for v in rows))
 files=[O/'result.json',O/'actual_resource_host_exit_v1.json',O/'actual_CPU_recovery_exit_v1.json',B/'actual_host_wait_exit.json',B/'campaign_failure.json',ROOT/'protocol.json',ROOT/'attempt/barrier/normal_child_exit.json',ROOT/'attempt/resources.jsonl']+list((O/'audits').rglob('*.json'))
 value=dict(passed=True,complete=True,qualification_kind='complete_normal_GPU_child_then_separate_normal_CPU_audit_v1',root=str(ROOT),protocol_identity=spec['identity'],actual_original_outer_host_exit_code=1,original_failure_preserved=True,actual_GPU_child_exit_code=0,actual_CPU_recovery_exit_code=0,no_new_GPU_work_or_episodes=True,no_failed_partial_branches_promoted=True,whole_native_branches=36,primary_and_zero_shadow_episodes=32,full_whole_controls_exact=True,resource180_raw_exact=True,checkpoint_sha256=spec['candidate_checkpoint_sha256'],groups=groups,total_functional_GPU_seconds_including_old_failures=4225.221,functional_GPU_budget=7200,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')},input_sha256={str(p):sha(p)for p in files})
 write(O/'qualified_TRAIN/qualification.json',value);print('SPLIT_NORMAL_GPU_AND_CPU_QUALIFIED_WITH_OLD_HOST1_PRESERVED',groups)
if __name__=='__main__':main()
