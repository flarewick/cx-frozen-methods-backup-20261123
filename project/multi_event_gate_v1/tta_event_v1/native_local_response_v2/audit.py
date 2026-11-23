"""Independent25 frozen solves, causal FIT partitions, donors and saved predictions."""
from pathlib import Path
import importlib.util,numpy as np
C=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('actual_structured_local_response_v2_study',C/'study.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
def run():
 p,x,a,meta=s.dataset();D=s.D;plan=s.read(D/'protocol.json');report=s.read(D/'report.json');pred=s.load(D/'predictions.npz')
 assert report['complete']and report['models']==25;checks=1;largest=0.;models=[]
 for group in('source_sha256','input_sha256'):
  for path,h in plan[group].items():assert s.sha(path)==h;checks+=1
 for mode in s.MODES:
  for fold in range(5):
   f=s.read(D/(mode+'_fold%d_freeze.json'%fold));file=D/(mode+'_fold%d.npz'%fold);st=s.load(file);assert s.sha(file)==f['model_sha256'];checks+=1
   tr=np.array(f['train']);te=np.array(f['test'])
   assert all(meta[i]['segment']=='origin'and meta[i]['available_before_following_query']and meta[i]['seed']%5!=fold for i in tr);checks+=1
   assert all(meta[i]['seed']%5==fold for i in te);checks+=1
   assert not{meta[i]['seed']for i in tr}&{meta[i]['seed']for i in te};checks+=1
   xx,v,kind,gate,donors=s.inputs(x,a,meta,tr,te,mode);assert donors==f['donors'];checks+=1
   for name,ids in(('train',tr),('test',te)):
    for i,j in zip(ids,donors.get(name,[])):
     assert j in tr and meta[j]['seed']!=meta[i]['seed']and meta[j]['task']==meta[i]['task']and meta[j]['object_index']==meta[i]['object_index'];checks+=1
   for got,expected in((st['train_context'],xx[tr]),(st['train_action'],v[tr]),(st['train_kind'],kind[tr])):
    assert np.array_equal(got,expected);checks+=1
   assert st['context_scale'][43]==st['context_scale'][52]==1. and st['ridge']==1.;checks+=1
   assert np.all(v[:,38:40]==0);checks+=1
   if mode=='without_history':assert np.all(xx[:,34:44]==0)and np.all(v[:,192:]==0);checks+=1
   if mode=='history_only':assert np.all(v[:,:192]==0);checks+=1
   if mode=='without_locality':assert np.all(gate==1);checks+=1
   w,_=s.metrics.loss_weights(a['target'][tr],[meta[i]for i in tr]);assert np.array_equal(w,st['fit_weight']);checks+=1
   K=s.m.kernel(st,xx[tr],v[tr],kind[tr]);A=K*w[:,None]*w[None];norm=float(np.trace(A)/len(tr))
   assert abs(norm-float(st['normalizer']))<1e-14;checks+=1
   A+=norm*np.eye(len(tr));rhs=a['target'][tr]*w[:,None]
   res=A@(st['weights']/w[:,None])-rhs;error=float(np.linalg.norm(res)/max(float(np.linalg.norm(rhs)),1e-30));largest=max(largest,error)
   assert error<1e-8;checks+=1
   assert np.array_equal(s.m.predict(st,xx[te],v[te],kind[te]),pred[mode][te]);checks+=1
   # Physical null in same frozen model: no drive/free motion cannot emit predicted movement.
   zero=np.zeros((len(te),194));assert np.array_equal(s.m.predict(st,xx[te],zero,kind[te]),np.zeros((len(te),2)));checks+=1
   models.append(dict(mode=mode,fold=fold,normal_equation_relative_residual=error,train_rows=len(tr),test_rows=len(te)))
 for segment in('origin','following'):
  ids=np.array([i for i,r in enumerate(meta)if r['segment']==segment]);mm=[meta[i]for i in ids]
  for mode in s.MODES:
   got,_=s.metrics.case_errors(pred[mode][ids],a['target'][ids],mm)
   assert got==report['results'][segment]['scores'][mode]['summary'];checks+=1
 s.write(D/'independent_audit.json',dict(passed=True,checks=checks,models=25,rows=len(meta),model_checks=models,
  largest_normal_equation_relative_residual=largest,all_frozen_models_exact_physical_null_verified=True,
  origin_only_FIT_whole_seed_groups_and_FIT_donors_verified=True,metrics_and_predictions_exact=True,
  source_sha256=s.sha(Path(__file__)),report_sha256=s.sha(D/'report.json'),diagnostic_support_gate_passed=report['diagnostic_support_gate_passed'],
  online_updates=0,TTA_utility_proven=False))
 print('STRUCTURED25_FROZEN_RESPONSE_INDEPENDENT_AUDIT_PASS',checks,largest,flush=True)
if __name__=='__main__':run()
