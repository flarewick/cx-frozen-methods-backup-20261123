"""Read only qualified whole batches; paired counts and exact discordance test."""
from .common import O,TASKS,read,replace,sha
import math
from collections import Counter

def summarize(final=False):
 records=[];batches=[];strata=[];missing={t:{c:0 for c in ('baseline','M_event_full','TTA')}for t in TASKS}
 from .host import qualified_roots
 roots=qualified_roots(read(O/'study.json'))
 for index,root in sorted(roots.items()):
  file=root/'terminal_audit.json'
  if not file.exists():continue
  terminal=read(file);assert terminal['passed']
  for p,digest in terminal['input_sha256'].items():
   from .common import mapped
   assert sha(mapped(p))==digest
  audit=read(root/'CPU_episode_audit.json');records.extend(audit['records']);batches.append(dict(batch_index=index,root=str(root)))
  strata.append(dict(batch_index=index,study_version=read(root/'protocol.json')['version'],projection_guard=audit.get('projection_guard_version','original_strict_projection_no_exception_in_qualified_batch'),endpoint_readback=audit.get('endpoint_readback_version','original_endpoint_order_no_conflict_in_qualified_batch'),endpoint_refreshes=len(audit.get('endpoint_readback_evidence',[])),endpoint_RGB_changes=sum(w['RGB_changed']for w in audit.get('endpoint_readback_evidence',[]))))
  for entry in audit.get('projection_source_evidence',[]):
   source_path=__import__('pathlib').Path(entry['file']);task=source_path.parents[2].name.split(':')[0];condition=source_path.parents[1].name
   if task in missing and condition in missing[task]:missing[task][condition]+=entry['unsupported_windows']
 groups={}
 for task in TASKS:
  rows=sorted([r for r in records if r['task']==task],key=lambda r:r['episode_index']);assert [r['episode_index']for r in rows]==list(range(len(rows)))and len({r['seed']for r in rows})==len(rows)
  full=sum(r['full']for r in rows);tta=sum(r['TTA']for r in rows);rescue=[r['seed']for r in rows if r['rescue']];harm=[r['seed']for r in rows if r['harm']];discordant=len(rescue)+len(harm);tail=sum(math.comb(discordant,k)for k in range(min(len(rescue),len(harm))+1));p=min(1.,2*tail/(2**discordant))if discordant else 1.
  baseline=sum(r['baseline']for r in rows)
  groups[task]=dict(baseline_successes=baseline,baseline_success_rate=baseline/len(rows)if rows else None,completed=len(rows),requested=100,full_successes=full,TTA_successes=tta,paired_net_successes=tta-full,difference_percentage_points=(tta-full)*100/len(rows)if rows else None,full_success_rate=full/len(rows)if rows else None,TTA_success_rate=tta/len(rows)if rows else None,rescued_seeds=rescue,harmed_seeds=harm,exact_two_sided_discordance_p=p,valid_seeds=[r['seed']for r in rows])
 for task in TASKS:
  rows=[r for r in records if r['task']==task];comparisons={}
  for left,right in [('baseline','full'),('full','TTA'),('baseline','TTA')]:
   gains=[r['seed']for r in rows if not r[left]and r[right]];losses=[r['seed']for r in rows if r[left]and not r[right]]
   d=len(gains)+len(losses);p=min(1.,2*sum(math.comb(d,k)for k in range(min(len(gains),len(losses))+1))/(2**d))if d else 1.
   comparisons[left+'_to_'+right]=dict(net_successes=len(gains)-len(losses),rescued_seeds=gains,harmed_seeds=losses,exact_two_sided_discordance_p=p)
  groups[task]['paired_comparisons']=comparisons
 for task in TASKS:
  selected=[r for r in records if r['task']==task];tta=[next(b for b in r['branches']if b['condition']=='TTA')for r in selected]
  groups[task]['TTA_coverage']=dict(denominator_all_completed=len(selected),semantic_available_cases=sum(b['semantic_available']for b in tta),cases_with_feedback_packet=sum(b['consumed_packets']>0 for b in tta),cases_with_accepted_update=sum(b['accepted_updates']>0 for b in tta),cases_with_applied_adaptation=sum(bool(b['applied_queries'])for b in tta),cases_with_actual_action_change=sum(r['first_changed_query']is not None for r in selected),semantic_unavailable_reasons=dict(Counter(b['semantic_unavailable_reason']for b in tta if not b['semantic_available'])))
 complete=all(g['completed']==100 for g in groups.values());assert not final or complete
 value=dict(complete=complete,implementation_strata=strata,projection_missing_windows=missing,qualified_batches=batches,groups=groups,checkpoint_sha256=read(O/'retained_candidate.json')['checkpoint_sha256'],official_seed_group=0,actual_seed_start=100000,actor_sampler_seed=17,strict_same_scene_initial_RNG_and_common_prefix=True,all_sources_models_and_update_rules_frozen=True,frozen_method_identity=read(O/'study.json')['frozen_method_identity'],category_interface_extension_disclosed=True,new_task_matched_full_controls=True,three_way_baseline_full_TTA=True,cross_task_transfer=True,perception_interface='native_transfer_semantics_v2',training_tasks_unchanged=['stack_blocks_two','stack_bowls_two'],interim_success_rates_not_final_efficacy=not complete)
 replace(O/('complete_comparison.json'if complete else'progress.json'),value);return value
if __name__=='__main__':
 import json
 print(json.dumps(summarize(),ensure_ascii=False,indent=2))
