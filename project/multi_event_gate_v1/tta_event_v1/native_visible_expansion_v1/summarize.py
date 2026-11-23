"""All fixed16 developer identities, paired scene statistics and explicit missingness."""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib,math,random
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_visible_expansion_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def verify_terminal(value):
 assert value['passed']and value['actual_host_exit_code']==value['actual_child_exit_code']==0 and value['own_GPU_released']
 canonical=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA');P=T.parents[1]
 for file,digest in value['input_sha256'].items():
  p=Path(file)
  if not p.exists()and p.is_relative_to(canonical):p=P/p.relative_to(canonical)
  assert sha(p)==digest,('Terminal evidence changed',str(p))
def percentile(x,p):
 x=sorted(x);i=(len(x)-1)*p;l=int(i);return x[l]+(x[min(l+1,len(x)-1)]-x[l])*(i-l)
def compare(rows,condition):
 requested=len(rows);valid=[r for r in rows if r['initialized']];missing=[r['case_id']for r in rows if not r['initialized']]
 if not valid:return dict(requested=requested,initialized=0,missing=missing,estimate=None)
 rescues=sum(not r['outcomes']['M_event_full']and r['outcomes'][condition]for r in valid);harms=sum(r['outcomes']['M_event_full']and not r['outcomes'][condition]for r in valid)
 full=sum(r['outcomes']['M_event_full']for r in valid);tta=sum(r['outcomes'][condition]for r in valid);discordant=rescues+harms
 p=min(1.,2*sum(math.comb(discordant,k)for k in range(min(rescues,harms)+1))/2**discordant)if discordant else 1.
 groups=defaultdict(list)
 for r in valid:groups[r['seed']].append(int(r['outcomes'][condition])-int(r['outcomes']['M_event_full']))
 keys=sorted(groups);rng=random.Random(17);samples=[]
 for _ in range(2000):
  values=[v for key in rng.choices(keys,k=len(keys))for v in groups[key]];samples.append(sum(values)/len(values))
 return dict(requested=requested,initialized=len(valid),missing=missing,full_successes=full,candidate_successes=tta,rescues=rescues,harms=harms,paired_net_rate=(rescues-harms)/len(valid),whole_seed_groups=len(groups),whole_numeric_seed_bootstrap2000_percentile95=[percentile(samples,.025),percentile(samples,.975)],exact_two_sided_discordance_p=p,requested_population_net_rate_missing_bounds=[(rescues-harms-len(missing))/requested,(rescues-harms+len(missing))/requested],interpretation='Developer comparison only. Missing outcomes unknown; no independent efficacy or stability claim.')
def summarize():
 design=read(O/'design.json');rows=[];inputs={str(O/'design.json'):sha(O/'design.json')}
 oldfile=T/'outputs/native_visible_runtime_v1/terminal_audit.json';old=read(oldfile);verify_terminal(old);inputs[str(oldfile)]=sha(oldfile)
 oldaudit=Path(old['root'])/'CPU_episode_audit.json'
 for record in read(oldaudit)['records']:
  outcomes={b['condition']:b['success']for b in record['branches']};rows.append(dict(case_id=record['case_id'],task=record['case_id'].split(':')[0],seed=int(record['case_id'].split(':')[1]),cohort='old4',initialized=True,gpu_uuid='GPU-0f89dea0-728e-1d23-fd62-d25b9bad3e4f',outcomes=outcomes,original_full_zero_qualification_preserved=True))
 inputs[str(oldaudit)]=sha(oldaudit)
 for letter in 'abc':
  base=T/'outputs'/('native_visible_expand_'+letter+'_v1');terminal=read(base/'terminal_audit.json');verify_terminal(terminal)
  spec=read(base/'protocol.json');assert spec['expanded_design_identity']==design['identity']and spec['checkpoint_sha256']==design['checkpoint_sha256']
  auditfile=Path(terminal['root'])/'CPU_episode_audit.json';audit=read(auditfile);assert audit['passed']and audit['all_requested_complete']
  for record in audit['records']:
   outcomes={b['condition']:b['success']for b in record['branches']};assert list(outcomes)==design['conditions']
   if record['initialized']:assert all(type(v)is bool for v in outcomes.values())
   else:assert all(v is None for v in outcomes.values())
   rows.append(dict(case_id=record['case_id'],task=record['case_id'].split(':')[0],seed=int(record['case_id'].split(':')[1]),cohort=letter,initialized=record['initialized'],gpu_uuid=spec['required_gpu_uuid'],outcomes=outcomes))
  for p in(base/'terminal_audit.json',base/'protocol.json',auditfile):inputs[str(p)]=sha(p)
 assert len(rows)==len({r['case_id']for r in rows})==16 and {r['case_id']for r in rows}=={r['case_id']for r in design['scenes']}
 groups={}
 populations={'all16':rows,'old4':[r for r in rows if r['cohort']=='old4'],'new12':[r for r in rows if r['cohort']!='old4']}
 populations.update({task:[r for r in rows if r['task']==task]for task in ('stack_blocks_two','stack_bowls_two')})
 populations.update({'hardware_'+gpu:[r for r in rows if r['gpu_uuid']==gpu]for gpu in {r['gpu_uuid']for r in rows}})
 for key,pop in populations.items():groups[key]={condition:compare(pop,condition)for condition in('TTA','wrong_feedback')}
 value=dict(complete=True,requested_scenes=16,requested_comparison_branches=80,actual_complete_scene_bundles=sum(r['initialized']for r in rows),missing_initializations=[r['case_id']for r in rows if not r['initialized']],candidate_checkpoint_sha256=design['checkpoint_sha256'],old4_new12_separate=True,rows=rows,groups=groups,input_sha256=inputs,source_sha256={str(Path(__file__)):sha(__file__)},new_optimizer_steps=0,new_independent_scenes=0,TTA_utility_proven=False,independent_confirmation_still_required=True)
 dest=O/'complete_comparison.json'
 if dest.exists():assert read(dest)==value
 else:write(dest,value)
 return value
if __name__=='__main__':print(json.dumps(summarize()['groups'],indent=2),flush=True)
