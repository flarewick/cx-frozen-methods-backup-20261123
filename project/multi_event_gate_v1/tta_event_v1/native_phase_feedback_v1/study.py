"""Frozen phase-only auxiliary FIT and single-use packet controls; no online TTA."""
from pathlib import Path
import importlib.util,numpy as np
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_phase_feedback_v1';D=O/'study'
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
old=module('source_structured_native_response_v2',T/'native_local_response_v2/study.py')
read,load,sha,write=old.read,old.load,old.sha,old.write;metrics=old.metrics;m=old.m;MODES=old.MODES
def dataset():
 p,x,a,meta=old.dataset();phase=read(O/'eligibility.json');allocation=read(O/'packet_allocation.json')
 assert read(O/'queue_checks_v2.json')['passed']and read(O/'actual_query_replay.json')['passed']
 assert read(T/'outputs/native_local_response_v2/study/independent_audit.json')['passed']
 assert len(meta)==len(phase['rows'])==allocation['all_rows']
 eligible=np.array([r['measurement']['available']for r in phase['rows']],bool)
 for i,r in enumerate(phase['rows']):
  assert r['row']==i and r['case_id']==meta[i]['case_id']and r['start_tick']==meta[i]['start_tick']and r['end_tick']==meta[i]['end_tick']
 return p,x,a,meta,eligible,allocation
def summarize(pred,y,meta,ids):
 mm=[meta[i]for i in ids];yy=y[ids]
 scores={mode:metrics.summary(v[ids],yy,mm)for mode,v in pred.items()}
 controls={mode:old.old.compare(pred['correct'][ids],pred[mode][ids],yy,mm)for mode in MODES[1:]}
 return dict(rows=len(ids),scores=scores,paired_controls=controls)
def run():
 p,x,a,meta,eligible,allocation=dataset();D.mkdir(parents=True,exist_ok=True)
 sources=[C/n for n in('study.py','feedback_queue.py','eligibility.py','replay.py','checks.py')]+[T/'native_local_response_v2/study.py',T/'native_local_response_v2/model.py']
 inputs=[O/n for n in('eligibility.json','packet_allocation.json','queue_checks_v2.json','actual_query_replay.json')]+[T/'outputs/native_local_response_v2'/n for n in('features_protocol.json','features.npz','checks.json','study/independent_audit.json')]
 plan=dict(modes=MODES,ridge=1.,models=25,all_rows=len(meta),phase_FIT_eligible_rows=int(eligible.sum()),
  primary_actual_q1_rows=len(allocation['actual_q1_rows']),provisional_q2_rows=len(allocation['hypothetical_q2_rows']),
  FIT='Only actual post-own95 windows, endpoint lag>=.25, original32receipt complete. Other TRAIN seeds may include both original/following actual physics; allnumericseed/CZ/objects overlap heldout%5.',
  timing='Frame/command observations remain attributed to their actual segment. q1 can use only450 early rows;887 q2 rows require a NOT YET recorded third nativequery. Once-per-source budget excludes1823 later rows from update-loss packets.',
  model='Exact frozen native_local_response_v2 support/free-dynamics kernel and ridge1; same53 context/197actualcommands; no hyperparameter, threshold or capacity search.',
  scoring='All3164 stage measurements and single-use1337 packetunion; q1/q2 separately; retained1423 pre/tooearly windows also fully reported. Relative target only both observed. Numericseedbootstrap2000.',
  passing='Allphase and packetunion correct each task positive/lower95>0 and advantages over wrong/historyonly lower95>0; q1/q2 each task positive; packetunion relative eachtask positive/lower95>0. Only developer auxiliary support, not deployed complete candidate/independent/TTA task evidence.',
  full_online_candidate_frozen=False,actual_third_query_collected=False,
  source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for p in inputs},
  online_updates=0,direction_training=False,new_independent_scenes=0,TTA_utility_proven=False)
 write(D/'protocol.json',plan);pred={};logs={}
 for mode in MODES:
  output=np.full(a['target'].shape,np.nan);folds=[]
  for fold in range(5):
   tr=np.array([i for i,r in enumerate(meta)if eligible[i]and r['seed']%5!=fold])
   te=np.array([i for i,r in enumerate(meta)if r['seed']%5==fold])
   assert len(tr)and len(te)and not{meta[i]['seed']for i in tr}&{meta[i]['seed']for i in te}
   xx,v,kind,gate,donors=old.inputs(x,a,meta,tr,te,mode)
   state,scales=old.fit(xx[tr],v[tr],kind[tr],a['target'][tr],[meta[i]for i in tr])
   file=D/(mode+'_fold%d.npz'%fold)
   with file.open('xb')as f:np.savez_compressed(f,**state)
   frozen=dict(fold=fold,train=tr.tolist(),test=te.tolist(),donors=donors,model_sha256=sha(file),FIT_post95_only=True,train_target_scales=scales)
   write(D/(mode+'_fold%d_freeze.json'%fold),frozen);folds.append(frozen)
   output[te]=m.predict(state,xx[te],v[te],kind[te])
   print('PHASE_LOCAL_RESPONSE_FOLD',mode,fold,len(tr),len(te),flush=True)
  assert np.isfinite(output).all();pred[mode]=output;logs[mode]=folds
 with(D/'predictions.npz').open('xb')as f:np.savez_compressed(f,**pred)
 write(D/'all_models_frozen.json',dict(complete=True,models=25,all_models_fixed_before_heldout_metrics=True,folds=logs,
  protocol_sha256=sha(D/'protocol.json'),predictions_sha256=sha(D/'predictions.npz')))
 q1=np.array(allocation['actual_q1_rows'],int);q2=np.array(allocation['hypothetical_q2_rows'],int);union=np.r_[q1,q2]
 strata=dict(all_postphase=np.flatnonzero(eligible),single_use_packet_union=union,actual_q1=q1,hypothetical_q2=q2,
   retained_prephase_or_shortlag=np.flatnonzero(~eligible))
 results={};passing=True
 bykey={(r['case_id'],r['branch'],r['start_tick'],r['end_tick'],r['object_index']):i for i,r in enumerate(meta)}
 for name,ids in strata.items():
  result=summarize(pred,a['target'],meta,ids);membership=set(ids.tolist())
  rel=[i for i in ids if meta[i]['object_index']==0 and a['relative_available'][i]
    and bykey[(meta[i]['case_id'],meta[i]['branch'],meta[i]['start_tick'],meta[i]['end_tick'],1)]in membership]
  mate=[bykey[(meta[i]['case_id'],meta[i]['branch'],meta[i]['start_tick'],meta[i]['end_tick'],1)]for i in rel]
  result['relative_windows']=len(rel);result['relative_scores']={mode:metrics.summary(pred[mode][rel]-pred[mode][mate],a['relative_target'][rel],[dict(meta[i],object_index=-1)for i in rel])for mode in MODES}
  results[name]=result;correct=result['scores']['correct']
  if name in('all_postphase','single_use_packet_union'):
   passing&=correct['whole_seed_bootstrap95'][0]>0 and all(correct['summary'][task]['improvement']>0 for task in('stack_blocks_two','stack_bowls_two'))
   passing&=all(result['paired_controls'][mode]['whole_seed_bootstrap95'][0]>0 for mode in('wrong_command','history_only'))
  if name in('actual_q1','hypothetical_q2'):
   passing&=all(correct['summary'][task]['improvement']>0 for task in('stack_blocks_two','stack_bowls_two'))
  if name=='single_use_packet_union':
   rr=result['relative_scores']['correct'];passing&=rr['whole_seed_bootstrap95'][0]>0 and all(rr['summary'][task]['improvement']>0 for task in('stack_blocks_two','stack_bowls_two'))
  print('PHASE_LOCAL_RESPONSE_RESULT',name,{mode:v['summary']for mode,v in result['scores'].items()},flush=True)
 for group in('source_sha256','input_sha256'):
  for path,h in plan[group].items():assert sha(path)==h
 write(D/'report.json',dict(complete=True,results=results,developer_auxiliary_support_gate_passed=bool(passing),models=25,all_rows=len(meta),
  original_requested_cases=32,missing_cases=p['missing_cases'],all_models_frozen_sha256=sha(D/'all_models_frozen.json'),
  predictions_sha256=sha(D/'predictions.npz'),full_online_candidate_frozen=False,actual_third_query_collected=False,
  online_updates=0,direction_training=False,new_independent_scenes=0,TTA_utility_proven=False))
 print('PHASE_LOCAL_RESPONSE_COMPLETE',bool(passing),flush=True)
if __name__=='__main__':run()
