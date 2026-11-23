"""Verified TRAIN-only feature cache from full audited cohorts; no holdout access."""
import json,os,sys
from pathlib import Path
import numpy as np,torch
import benefit_followup_protocol_v2 as p
import benefit_followup_learning_v6 as old
from benefit_followup_stream_v3 import atomic
OUT=p.HOME/'TRAIN472_diagnosis_v106'
def isolation():
 old.isolation();blocked=[p.HOME/'supervision_growth_v68',p.HOME/'sampler_followup_v53/prospective_validation_v57',p.HOME/'supplement_v24/validation_v29',p.HOME/'cohort_ensemble_v44/validation',p.W/'outputs/task_adapter_eval',p.HOME/'supervision_growth_replacement_v84/prospective_validation_v94'];blocked=[q.resolve()for q in blocked]
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
   q=Path(os.fsdecode(args[0])).resolve()
   if any(q==v or v in q.parents for v in blocked):raise RuntimeError('Quarantined data or DEV/TEST forbidden')
 sys.addaudithook(guard)
 for q in blocked:
  try:(q/'forbidden_probe').open()
  except RuntimeError:pass
  else:raise AssertionError('Isolation failed')
def main():
 torch.set_num_threads(1);assert not list(Path('/dev').glob('nvidia*'))and not torch.cuda.is_initialized();isolation();OUT.mkdir(exist_ok=True);assert not(OUT/'TRAIN472_features.json').exists();rng=torch.get_rng_state().clone();cache=p.HOME/'failure_diagnosis_v66/TRAIN392_features.json';groups=json.loads(cache.read_text())['groups'];audit={}
 for block in ('train1','train2'):
  root=p.HOME/('supervision_growth_replacement_v84'if block=='train1'else'supervision_growth_canonical_v96')/block
  for n in ['collection_audit.json','mixed_membership_final_audit.json','supplement_full_audit_receipt_v27.json']:
   v=json.loads((root/n).read_text());assert v['passed']and v['complete'];audit[str(root/n)]=p.sha(root/n)
  assert json.loads((root/'collection_audit.json').read_text())['resource_audit']['passed'];groups.append(old.load(root,'TRAIN','supervision68_'+block))
 rows=sum(groups,[]);assert len(groups)==12 and len(rows)==472 and len({(r['task'],r['seed'])for r in rows})==472
 for r in rows:
  if r['features']is not None:r['features']={k:np.asarray(v,np.float64).tolist()for k,v in r['features'].items()}
 summary={}
 for t in sorted({r['task']for r in rows}):
  rr=[r for r in rows if r['task']==t];summary[t]=dict(scenes=len(rr),A0=sum(r['a']for r in rr),B=sum(r['b']for r in rr),classes={k:sum(r['label']==k for r in rr)for k in old.CLASSES})
 assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized();atomic(OUT/'TRAIN472_features.json',dict(passed=True,TRAIN_only=True,DEV_excluded=True,TEST_excluded=True,groups=groups,source_sha256=p.sha(__file__),prior_cache_sha256=p.sha(cache),audits=audit,adapter_sha256=p.sha(p.CANDIDATE)));atomic(OUT/'summary.json',dict(passed=True,TRAIN_only=True,tasks=summary));print('TRAIN472_FEATURE_CACHE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
