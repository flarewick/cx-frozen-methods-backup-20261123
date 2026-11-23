"""One structural candidate and fixed mechanism controls, origin-only whole-seed FIT."""
from pathlib import Path
import importlib.util,sys,numpy as np
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_local_response_v2';D=O/'study'
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
old=module('frozen_origin_response_v1_study',T/'native_origin_response_v1/study.py')
m=module('actual_local_support_response_v2_model',C/'model.py')
read,load,sha,write=old.read,old.load,old.sha,old.write;metrics=old.metrics
MODES=('correct','wrong_command','without_locality','without_history','history_only')
def dataset():
 p=read(O/'features_protocol.json');f=load(O/'features.npz');assert sha(O/'features.npz')==p['features_sha256']
 original,xx,a,meta=old.dataset();assert len(f['context'])==len(meta)and p['metadata']==original['metadata']
 assert read(O/'checks.json')['passed'];return p,f['context'],a,meta
def fit(x,v,kind,y,meta):
 s=m.state(x,v,kind);w,scales=metrics.loss_weights(y,meta);gram=m.kernel(s,x,v,kind)*w[:,None]*w[None]
 gram=(gram+gram.T)/2;eigen,vec=np.linalg.eigh(gram);assert eigen.min()>-max(1.,eigen.max())*1e-10
 norm=float(np.trace(gram)/len(y));assert norm>0;rhs=vec.T@(y*w[:,None])
 s.update(weights=w[:,None]*(vec@(rhs/(eigen[:,None]+norm))),ridge=np.asarray(1.),normalizer=np.asarray(norm),fit_weight=w)
 return s,scales
def inputs(x,a,meta,tr,te,mode):
 u,donors=old.inputs(x,a,meta,tr,te,'wrong_command'if mode=='wrong_command'else'correct')
 xx,v,kind,gate=m.coordinates(x,u,a['start_points'],mode)
 return xx,v,kind,gate,donors
def run():
 p,x,a,meta=dataset();D.mkdir(parents=True,exist_ok=True)
 sources=[C/n for n in('study.py','features.py','model.py','checks.py')]+[T/'native_origin_response_v1/study.py',T/'silhouette_geometry_state_v1/metrics.py']
 files=[O/n for n in('features_protocol.json','features.npz','checks.json')]+[Path(p['local_source'])/n for n in('protocol.json','data.npz','independent_audit.json')]
 protocol=dict(modes=MODES,ridge=1.,no_hyperparameter_grid=True,no_capacity_search=True,
  models=25,FIT='Only origin feedback windows, heldout allnumericseed%5. CZ/objects/overlap in same group. Following never FIT or train scale/donor.',
  local_support='exp(-.5*(min distance commandedTCP sweep to currentRGB corners/current visible radius)^2), independent arms. No GT/contact claim.',
  dynamics='Only arm-local motion/grip changes drive new response. Time/age removed from direct drive; observed past object velocity*actualdt is free dynamics.',
  history='Exact previous arrived adjacent window ending at currentstart, all4232 rebuilt from cropped actual RGB prefix. No current/future output.',
  controls='Wrong FIT-only trace, allsame old donor rules. withoutlocality/withouthistory/historyonly fixed mechanism controls; no favorable selector.',
  scoring='Origin/following separately, both perobject and fixed samewindow object0-minus1 relative, allmissing retained, numericseedbootstrap2000.',
  passing='Correct both tasks positive and lower95>0 for origin/following, wrong/historyonly pairedadvantages lower95>0; following relative both tasks positive/lower95>0. Developer support only, NOT complete deployed candidate/independent/TTA efficacy.',
  source_sha256={str(f):sha(f)for f in sources},input_sha256={str(f):sha(f)for f in files},online_updates=0,TTA_utility_proven=False)
 write(D/'protocol.json',protocol);pred={};logs={}
 for mode in MODES:
  out=np.full(a['target'].shape,np.nan);folds=[]
  for fold in range(5):
   tr=np.array([i for i,r in enumerate(meta)if r['seed']%5!=fold and r['segment']=='origin'and r['available_before_following_query']])
   te=np.array([i for i,r in enumerate(meta)if r['seed']%5==fold])
   assert not{meta[i]['seed']for i in tr}&{meta[i]['seed']for i in te}
   xx,v,kind,gate,donors=inputs(x,a,meta,tr,te,mode)
   state,scales=fit(xx[tr],v[tr],kind[tr],a['target'][tr],[meta[i]for i in tr])
   file=D/(mode+'_fold%d.npz'%fold)
   with file.open('xb')as f:np.savez_compressed(f,**state)
   frozen=dict(fold=fold,train=tr.tolist(),test=te.tolist(),donors=donors,FIT_following_rows=0,
    model_sha256=sha(file),train_target_scales=scales,mean_local_support=gate[tr].mean(0).tolist())
   write(D/(mode+'_fold%d_freeze.json'%fold),frozen);folds.append(frozen)
   out[te]=m.predict(state,xx[te],v[te],kind[te])
   print('STRUCTURED_LOCAL_RESPONSE_FOLD',mode,fold,len(tr),len(te),flush=True)
  assert np.isfinite(out).all();pred[mode]=out;logs[mode]=folds
 with(D/'predictions.npz').open('xb')as f:np.savez_compressed(f,**pred)
 write(D/'all_models_frozen.json',dict(complete=True,models=25,heldout_targets_not_fit_or_scored_before_all_models_frozen=True,
  folds=logs,predictions_sha256=sha(D/'predictions.npz'),protocol_sha256=sha(D/'protocol.json')))
 result={};passing=True
 bykey={(r['case_id'],r['branch'],r['start_tick'],r['end_tick'],r['object_index']):i for i,r in enumerate(meta)}
 for segment in('origin','following'):
  ids=np.array([i for i,r in enumerate(meta)if r['segment']==segment]);mm=[meta[i]for i in ids];yy=a['target'][ids]
  scores={mode:metrics.summary(pred[mode][ids],yy,mm)for mode in MODES}
  controls={mode:old.compare(pred['correct'][ids],pred[mode][ids],yy,mm)for mode in MODES[1:]}
  rel=[i for i in ids if meta[i]['object_index']==0 and a['relative_available'][i]]
  mate=[bykey[(meta[i]['case_id'],meta[i]['branch'],meta[i]['start_tick'],meta[i]['end_tick'],1)]for i in rel]
  relative={mode:metrics.summary(pred[mode][rel]-pred[mode][mate],a['relative_target'][rel],[dict(meta[i],object_index=-1)for i in rel])for mode in MODES}
  result[segment]=dict(rows=len(ids),relative_windows=len(rel),scores=scores,paired_controls=controls,relative_scores=relative)
  cr=scores['correct'];passing&=cr['whole_seed_bootstrap95'][0]>0 and all(cr['summary'][task]['improvement']>0 for task in('stack_blocks_two','stack_bowls_two'))
  passing&=all(controls[mode]['whole_seed_bootstrap95'][0]>0 for mode in('wrong_command','history_only'))
  if segment=='following':
   rr=relative['correct'];passing&=rr['whole_seed_bootstrap95'][0]>0 and all(rr['summary'][task]['improvement']>0 for task in('stack_blocks_two','stack_bowls_two'))
  print('STRUCTURED_LOCAL_RESPONSE_RESULT',segment,{mode:v['summary']for mode,v in scores.items()},flush=True)
 for group in('source_sha256','input_sha256'):
  for path,h in protocol[group].items():assert sha(path)==h
 write(D/'report.json',dict(complete=True,results=result,diagnostic_support_gate_passed=bool(passing),rows=len(meta),models=25,
  original_requested_cases=32,missing_cases=p['missing_cases'],predictions_sha256=sha(D/'predictions.npz'),
  all_models_frozen_sha256=sha(D/'all_models_frozen.json'),online_updates=0,direction_training=False,new_independent_scenes=0,TTA_utility_proven=False))
 print('STRUCTURED_LOCAL_RESPONSE_COMPLETE',bool(passing),flush=True)
if __name__=='__main__':run()
