"""Current-only legal raw observation/action features; audited TRAIN packets only."""
import json,os
from pathlib import Path
import numpy as np
from cx_tta.onset_observables import validate_packet
DIMS={'action145':145,'observation488':488,'current593':593}
def features(packet,context,family):
 p=validate_packet(packet);ctx=np.asarray(context,np.float32);assert ctx.shape==(8,)and np.isfinite(ctx).all();arm=[0,1,2,3,4,5,7,8,9,10,11,12];state=p['state'][0].astype(np.float64);ee=p['ee_pose'][0].astype(np.float64).ravel();base=p['base'][0].astype(np.float64);delta=p['candidate_delta'][0].astype(np.float64)*.5
 # Preserve signed correction components in four chronological bins; no future executed action or simulator state.
 b=base[:,arm].reshape(4,8,12).mean(1).ravel();d=delta[:,arm].reshape(4,8,12).mean(1).ravel();grip=base[:,[6,13]].reshape(4,8,2).mean(1).ravel();history=p['has_history'].astype(np.float64).ravel();obs=np.r_[p['visual'][0].astype(np.float64).ravel(),p['text'][0].astype(np.float64).ravel(),state,ee,ctx]
 if family=='action145':x=np.r_[state,ee,b,grip,d,ctx,history]
 elif family=='observation488':x=obs
 elif family=='current593':x=np.r_[obs,b,grip,d,history]
 else:raise ValueError('Unknown raw legal family')
 assert x.shape==(DIMS[family],)and np.isfinite(x).all();return x

def main():
 import torch
 import benefit_followup_protocol_v2 as p
 from benefit_TRAIN472_cache_v106 import isolation
 from benefit_growth_learning import packet_at
 from benefit_followup_stream_v3 import atomic
 torch.set_num_threads(1);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();isolation();out=p.HOME/'raw_features_TRAIN472_v108';out.mkdir(exist_ok=True);assert not(out/'features.json').exists();source=p.HOME/'TRAIN472_diagnosis_v106/TRAIN472_features.json';data=json.loads(source.read_text());assert data['TRAIN_only']and data['DEV_excluded']and data['TEST_excluded'];groups=data['groups'];items=json.loads((p.HOME/'supplement_v24/learning_v25/frozen_TRAIN_inputs.json').read_text())['TRAIN_inputs'];mapping={v['name']:(Path(v['root']),v['split'],v['audit_sha256'])for v in items}
 for name,root in [('sampler53_train5',p.HOME/'sampler_followup_v53/train5'),('supervision68_train1',p.HOME/'supervision_growth_replacement_v84/train1'),('supervision68_train2',p.HOME/'supervision_growth_canonical_v96/train2')]:mapping[name]=(root,'TRAIN',p.sha(root/'collection_audit.json'))
 receipt={};resolved=[];checks=0;rng=torch.get_rng_state().clone()
 for group in groups:
  name=group[0]['group'];root,split,expected=mapping[name];assert p.sha(root/'collection_audit.json')==expected;a=json.loads((root/'collection_audit.json').read_text());assert a['passed']and a['complete'];plan=json.loads((root/'collect_plan.json').read_text());cases={(c['task'],c['seed']):c for c in a['cases']if c['split']==split};assert set(cases)=={(r['task'],r['seed'])for r in group};receipt[str(root/'collection_audit.json')]=expected
  # Historical preparation plans may name an empty identity. Resolve only a UNIQUE
  # completed journal containing the exact predeclared cohort seeds; validate the
  # original legal feature bytes against the signed historical cache below.
  from benefit_followup_gate_v2 import features as original_features
  for job in plan['jobs']:
   if job['branch']!='A0':continue
   seeds={r['seed']for r in group if r['task']==job['task']};current=Path(job['run']);journal=json.loads((current/'journal.json').read_text())
   if seeds.issubset({e['seed']for e in journal['episodes']}):continue
   candidates=[]
   for file in root.glob('execution*/'+job['task']+'/A0/journal.json'):
    jj=json.loads(file.read_text());ee=jj['episodes'];result=file.parent/'result.json'
    if seeds.issubset({e['seed']for e in ee})and all(e['complete']for e in ee)and result.exists()and json.loads(result.read_text())['complete']:candidates.append(file)
   assert len(candidates)==1,('No unique historical committed journal',root,job['task'],len(candidates));file=candidates[0];resolved.append(dict(group=name,task=job['task'],old_run=str(current),actual_run=str(file.parent),journal_sha256=p.sha(file),manifest_sha256=p.sha(file.parent/'manifest.json'),fixed_seeds=sorted(seeds)));job['run']=str(file.parent)
  for row in group:
   case=cases[row['task'],row['seed']];assert row['a']==int(case['successes']['A0'])and row['b']==int(case['successes']['B']);v=packet_at(plan,case)
   if v is None:assert row['features']is None and row['a']==row['b'];continue
   assert row['features']is not None;packet,ctx=v
   for family,reference in row['features'].items():assert np.array_equal(original_features(packet,ctx,family),np.asarray(reference,np.float64)),('Historical cached feature mismatch',name,row['task'],row['seed'],family)
   before={k:x.tobytes()for k,x in packet.items()};ff={f:features(packet,ctx,f).tolist()for f in DIMS};assert before=={k:x.tobytes()for k,x in packet.items()};row['features']=ff;checks+=1
  print('RAW108_GROUP',name,len(group),flush=True)
 assert len(groups)==12 and sum(map(len,groups))==472 and torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized();atomic(out/'features.json',dict(passed=True,TRAIN_only=True,DEV_excluded=True,TEST_excluded=True,groups=groups,source_sha256=p.sha(__file__),input_cache_sha256=p.sha(source),audits=receipt,current_only_legal_inputs=True,adapter_frozen_sha256=p.sha(p.CANDIDATE),packet_immutability_checks=checks,legacy_plan_resolution=resolved,all_original_cached_features_reproduced_exactly=True,original_plans_unmodified=True));print('RAW108_CACHE_COMPLETE',checks,flush=True)
if __name__=='__main__':main()
