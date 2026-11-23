"""Independent phase-only25 frozen solves and single-use query allocation audit."""
from pathlib import Path
import importlib.util,numpy as np
C=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('actual_phase_response_study_for_audit',C/'study.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
def run():
 p,x,a,meta,eligible,allocation=s.dataset();D=s.D;plan=s.read(D/'protocol.json');report=s.read(D/'report.json');pred=s.load(D/'predictions.npz')
 assert report['complete']and report['models']==25;checks=1;largest=0.;models=[]
 for group in('source_sha256','input_sha256'):
  for path,h in plan[group].items():assert s.sha(path)==h;checks+=1
 early=set(allocation['actual_q1_rows']);late=set(allocation['hypothetical_q2_rows']);ignored=set(allocation['ignored_later_rows_after_one_source_consumed']);censor=set(allocation['query_censored_rows'])
 groups=(early,late,ignored,censor)
 assert all(not a&b for i,a in enumerate(groups)for b in groups[i+1:]);checks+=1
 assert set.union(*groups)==set(np.flatnonzero(eligible).tolist());checks+=1
 replay=s.read(s.O/'actual_query_replay.json');assert early==set(replay['released_row_indices']);checks+=1
 consumed={(r['case_id'],r['branch'])for r in replay['rows']if r['source_consumed_once']}
 assert all((meta[i]['case_id'],meta[i]['branch'])not in consumed for i in late);checks+=1
 assert all((meta[i]['case_id'],meta[i]['branch'])in consumed for i in ignored);checks+=1
 for mode in s.MODES:
  for fold in range(5):
   f=s.read(D/(mode+'_fold%d_freeze.json'%fold));file=D/(mode+'_fold%d.npz'%fold);st=s.load(file)
   assert s.sha(file)==f['model_sha256'];checks+=1
   tr=np.array(f['train']);te=np.array(f['test'])
   assert all(eligible[i]and meta[i]['seed']%5!=fold for i in tr);checks+=1
   assert all(meta[i]['seed']%5==fold for i in te);checks+=1
   assert not{meta[i]['seed']for i in tr}&{meta[i]['seed']for i in te};checks+=1
   xx,v,kind,gate,donors=s.old.inputs(x,a,meta,tr,te,mode);assert donors==f['donors'];checks+=1
   for name,ids in(('train',tr),('test',te)):
    for i,j in zip(ids,donors.get(name,[])):
     assert j in tr and eligible[j]and meta[j]['seed']!=meta[i]['seed']and meta[j]['task']==meta[i]['task']and meta[j]['object_index']==meta[i]['object_index'];checks+=1
   for got,expected in((st['train_context'],xx[tr]),(st['train_action'],v[tr]),(st['train_kind'],kind[tr])):
    assert np.array_equal(got,expected);checks+=1
   assert np.all(v[:,38:40]==0)and st['ridge']==1.;checks+=1
   w,_=s.metrics.loss_weights(a['target'][tr],[meta[i]for i in tr]);assert np.array_equal(w,st['fit_weight']);checks+=1
   K=s.m.kernel(st,xx[tr],v[tr],kind[tr]);A=K*w[:,None]*w[None];norm=float(np.trace(A)/len(tr))
   assert abs(norm-float(st['normalizer']))<1e-14;checks+=1
   A+=norm*np.eye(len(tr));rhs=a['target'][tr]*w[:,None]
   res=A@(st['weights']/w[:,None])-rhs;error=float(np.linalg.norm(res)/max(float(np.linalg.norm(rhs)),1e-30));largest=max(largest,error)
   assert error<1e-8;checks+=1
   assert np.array_equal(s.m.predict(st,xx[te],v[te],kind[te]),pred[mode][te]);checks+=1
   assert np.array_equal(s.m.predict(st,xx[te],np.zeros((len(te),194)),kind[te]),np.zeros((len(te),2)));checks+=1
   models.append(dict(mode=mode,fold=fold,normal_equation_relative_residual=error,train_rows=len(tr),test_rows=len(te)))
 strata=dict(all_postphase=np.flatnonzero(eligible),single_use_packet_union=np.array(sorted(early|late),int),
  actual_q1=np.array(sorted(early),int),hypothetical_q2=np.array(sorted(late),int),retained_prephase_or_shortlag=np.flatnonzero(~eligible))
 # Scores are order invariant; original union is q1 thenq2. Reproduce its actual order exactly.
 strata['single_use_packet_union']=np.r_[allocation['actual_q1_rows'],allocation['hypothetical_q2_rows']]
 for name,ids in strata.items():
  mm=[meta[i]for i in ids]
  for mode in s.MODES:
   got,_=s.metrics.case_errors(pred[mode][ids],a['target'][ids],mm)
   assert got==report['results'][name]['scores'][mode]['summary'];checks+=1
 value=dict(passed=True,checks=checks,models=25,rows=len(meta),largest_normal_equation_relative_residual=largest,
  model_checks=models,phase_only_FIT_whole_seed_partition_and_FIT_donors_verified=True,
  all_frozen_models_exact_physical_null=True,single_use_packet_allocation_and_metrics_exact=True,
  actual_recorded_q1_rows=len(early),hypothetical_NOT_observed_q2_rows=len(late),
  developer_auxiliary_support_gate_passed=report['developer_auxiliary_support_gate_passed'],
  full_online_candidate_frozen=False,actual_third_query_collected=False,
  source_sha256=s.sha(Path(__file__)),report_sha256=s.sha(D/'report.json'),online_updates=0,TTA_utility_proven=False)
 s.write(D/'independent_audit.json',value);print('PHASE25_FROZEN_FITS_ALLOCATION_INDEPENDENT_AUDIT_PASS',checks,largest,flush=True)
if __name__=='__main__':run()
