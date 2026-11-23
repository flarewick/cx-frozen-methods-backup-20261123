"""Read only qualified whole batches; paired counts and exact discordance test."""
from .common import O,TASKS,read,replace,sha
import math

def summarize(final=False):
 records=[];batches=[]
 from .host import qualified_roots
 roots=qualified_roots(read(O/'study.json'))
 for index,root in sorted(roots.items()):
  file=root/'terminal_audit.json'
  if not file.exists():continue
  terminal=read(file);assert terminal['passed']
  for p,digest in terminal['input_sha256'].items():
   from .common import mapped
   assert sha(mapped(p))==digest
  records.extend(read(root/'CPU_episode_audit.json')['records']);batches.append(dict(batch_index=index,root=str(root)))
 groups={}
 for task in TASKS:
  rows=sorted([r for r in records if r['task']==task],key=lambda r:r['episode_index']);assert [r['episode_index']for r in rows]==list(range(len(rows)))and len({r['seed']for r in rows})==len(rows)
  full=sum(r['full']for r in rows);tta=sum(r['TTA']for r in rows);rescue=[r['seed']for r in rows if r['rescue']];harm=[r['seed']for r in rows if r['harm']];discordant=len(rescue)+len(harm);tail=sum(math.comb(discordant,k)for k in range(min(len(rescue),len(harm))+1));p=min(1.,2*tail/(2**discordant))if discordant else 1.
  groups[task]=dict(completed=len(rows),requested=100,full_successes=full,TTA_successes=tta,paired_net_successes=tta-full,difference_percentage_points=(tta-full)*100/len(rows)if rows else None,full_success_rate=full/len(rows)if rows else None,TTA_success_rate=tta/len(rows)if rows else None,rescued_seeds=rescue,harmed_seeds=harm,exact_two_sided_discordance_p=p,valid_seeds=[r['seed']for r in rows])
 complete=all(g['completed']==100 for g in groups.values());assert not final or complete
 value=dict(complete=complete,qualified_batches=batches,groups=groups,checkpoint_sha256=read(O/'retained_candidate.json')['checkpoint_sha256'],official_seed_group=0,actual_seed_start=100000,actor_sampler_seed=17,strict_same_scene_initial_RNG_and_common_prefix=True,all_sources_models_and_update_rules_frozen=True,historical55_76_are_references_not_primary_controls=True,interim_success_rates_not_final_efficacy=True)
 replace(O/('complete_comparison.json'if complete else'progress.json'),value);return value
if __name__=='__main__':
 import json
 print(json.dumps(summarize(),ensure_ascii=False,indent=2))
