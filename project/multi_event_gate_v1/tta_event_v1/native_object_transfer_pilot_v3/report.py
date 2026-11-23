"""All native pilot outcomes retained; qualification is not an efficacy test."""
from .common import TASKS,read,write,sha

def summarize(root,spec):
 audit=read(root/'CPU_episode_audit.json');assert audit['passed']and read(root/'terminal_audit.json')['passed'];groups={}
 for task in TASKS:
  records=[r for r in audit['records']if r['task']==task];assert len(records)==2
  branches=[b for r in records for b in r['branches']if b['condition']=='TTA']
  groups[task]=dict(paired_scenes=len(records),baseline_successes=sum(r['baseline']for r in records),full_successes=sum(r['full']for r in records),TTA_successes=sum(r['TTA']for r in records),semantic_available_scenes=sum(b['semantic_available']for b in branches),packet_scenes=sum(b['consumed_packets']>0 for b in branches),accepted_update_scenes=sum(b['accepted_updates']>0 for b in branches),later_action_changed_scenes=sum(r['first_changed_query']is not None for r in records),all_records=records)
 qualified=all(g['semantic_available_scenes']>0 and g['packet_scenes']>0 and g['accepted_update_scenes']>0 and g['later_action_changed_scenes']>0 for g in groups.values())
 value=dict(complete=True,stage=spec['stage'],groups=groups,interface_qualified=qualified,actual_native_episodes=audit['actual_native_episodes'],source_and_terminal_sha256={str(p):sha(p)for p in(root/'protocol.json',root/'CPU_episode_audit.json',root/'terminal_audit.json')},whole_zero_and_shadow_controls_checked=True,resource_and_normal_exit_qualified=True,formal_EVAL_used=False,no_task_efficacy_claim=True)
 write(root/'pilot_report.json',value);return {k:v for k,v in value.items()if k!='groups'}

def aggregate(roots,study):
 reports=[read(root/'pilot_report.json')for root in roots];assert len(reports)==study['logical_batch_count']
 groups={}
 for task in TASKS:
  records=[r for report in reports for r in report['groups'][task]['all_records']];assert len(records)==study['episodes_per_task']
  fields=('paired_scenes','baseline_successes','full_successes','TTA_successes','semantic_available_scenes','packet_scenes','accepted_update_scenes','later_action_changed_scenes')
  groups[task]={k:sum(report['groups'][task][k]for report in reports)for k in fields};groups[task]['all_records']=records
 qualified=all(g['semantic_available_scenes']>0 and g['packet_scenes']>0 and g['accepted_update_scenes']>0 and g['later_action_changed_scenes']>0 for g in groups.values())
 value=dict(complete=True,stage=study['stage'],interface_qualified=qualified,groups=groups,actual_native_episodes=sum(r['actual_native_episodes']for r in reports),complete_qualified_batches=len(reports),all_real_normal_exits_and_resource_audits=True,original_zero_shadow_controls_exact=True,formal_EVAL_used=False,no_task_efficacy_claim=True,input_sha256={str(root/'pilot_report.json'):sha(root/'pilot_report.json')for root in roots})
 from .common import O
 write(O/'complete_qualification.json',value)
 return {k:v for k,v in value.items()if k!='groups'}
