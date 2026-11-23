"""Read-only zero-tolerance diagnosis of inactive clutter attribute snapshots."""
import copy,gzip,json,hashlib
from pathlib import Path
from collections import Counter
from benefit_growth_protocol import ROOT,D,atomic,sha
from evaluate import host
EXPECTED=Counter({'063_tabletrashbin':1,'garbage':5,'ground':1,'wall':1,'table':1,'011_dustbin':1})
def read(path):
 with gzip.open(path,'rt')as f:return [json.loads(s)for s in f]
def episode(job,seed):
 ep=json.loads((host(job['run'])/'concurrency_queue.json').read_text())['completed'][str(seed)];session=host(job['run'])/'sessions'/ep['process_session'];stem=f"episode{ep['episode_index']:03d}_seed{seed}"
 path=session/'pilot_traces'/(stem+'.jsonl.gz');return ep,path,read(path)
def prove_inactive(rows):
 poses=[];snapshots=0
 for row in rows:
  native=[row['offline_only']['physical_before'],row['offline_only']['physical_after']]
  for snap in native:
   assert snap['usable']and not snap['coverage']['errors']
   assert Counter(x['name']for x in snap['actors'])==EXPECTED
   assert len(snap['articulations'])==1
   snapshots+=1
  for action in row['offline_only']['actions']:
   for side in ('before','after'):
    poses.append(action['offline_only'][side]['state']['offline_only']['object_poses']['cluttered_obj'])
 assert poses and all(x==poses[0]for x in poses),'Alias must be constant throughout episode'
 ghost=poses[0]
 for row in rows:
  for key in ('physical_before','physical_after'):
   snap=row['offline_only'][key];active=[x['pose']for x in snap['actors']]+[x['root_pose']for x in snap['articulations']]
   assert ghost not in active,'Do not exclude a current-scene object'
 return dict(pose=ghost,native_snapshots=snapshots,action_snapshots=len(poses),native_inventory_exact_no_clutter=True,alias_absent_from_every_current_native_snapshot=True,alias_constant_all_recorded_actions=True)
def cleaned(row):
 value=copy.deepcopy(row)
 for action in value['offline_only']['actions']:
  for side in ('before','after'):del action['offline_only'][side]['state']['offline_only']['object_poses']['cluttered_obj']
 return value

def main():
 report=json.loads((ROOT/'collection_audit.json').read_text());assert not report['passed']
 jobs={(j['task'],j['branch']):j for j in json.loads((ROOT/'collect_plan.json').read_text())['jobs']}
 cases=[];sources={str(ROOT/'collection_audit.json'):sha(ROOT/'collection_audit.json')}
 for c in report['cases']:
  if not(c['aa_differences']or c['preintervention_differences']):continue
  assert c['task']=='dump_bin_bigbin'
  for branch,issues in [('A1',c['aa_differences']),('B',c['preintervention_differences'])]:
   if not issues:continue
   assert all(x[0]=='full_passive_trace'for x in issues)
   a,pa,ra=episode(jobs[c['task'],'A0'],c['seed']);b,pb,rb=episode(jobs[c['task'],branch],c['seed'])
   da,db=prove_inactive(ra),prove_inactive(rb);sources[str(pa)]=sha(pa);sources[str(pb)]=sha(pb)
   count=len(ra)if branch=='A1'else c['single_event_query'];assert count>0
   if branch=='A1':assert len(ra)==len(rb)
   raw=[]
   for q in range(count):
    if ra[q]!=rb[q]:raw.append(q)
    assert cleaned(ra[q])==cleaned(rb[q]),(c['seed'],branch,q,'Other trace differences must remain rejected')
   assert raw==[x[1]for x in issues]
   cases.append(dict(task=c['task'],seed=c['seed'],branch=branch,raw_different_chunks=raw,reference_alias=da,candidate_alias=db,after_removing_only_proven_inactive_attribute_all_remaining_fields_exact=True))
 assert len(cases)==4
 for rel in ['multi_event_gate_v1/cx_tta/recording.py','multi_event_gate_v1/cx_tta/attribution_physics.py']:
  p=D.parent/rel;sources[str(p)]=sha(p)
 proof=dict(passed=True,diagnosis_only=True,scientific_acceptance=False,original_audit_still_failed=True,raw_data_unmodified=True,no_model_fit=True,numeric_tolerance=0,policy_inputs_actions_and_native_snapshots_unchanged=True,cases=cases,source_sha256=sources,checker_sha256=sha(__file__),conclusion='Only task-attribute object_poses/cluttered_obj in per-action diagnostics differs; values are constant and absent from every exhaustive native scene snapshot. Source object_poses enumerates attrs, while original setup can retain prior cluttered_obj attribute; do not accept until a separately documented strict validation amendment is implemented.')
 atomic(ROOT/'inactive_alias_diagnosis.json',proof);print('INACTIVE_ALIAS_DIAGNOSIS',json.dumps({k:v for k,v in proof.items()if k not in ('source_sha256','cases')}),flush=True)
if __name__=='__main__':main()
