"""Fixed independent physical-feedback gate, no fitting or candidate selection."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
from .paths import TTA,OUTPUT,read,write,digest,verify
from .contract import load_journal,load_packet,BRANCHES
from native_delayed_query_v1.predictor import FrozenPredictor
from native_phase_feedback_v1.eligibility import qualify
TASKS=('stack_blocks_two','stack_bowls_two')
MODES=('correct','wrong_command','without_locality','without_history','history_only')
ANALYSIS_IMPLEMENTED=True
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def controls():
 d=TTA/'outputs/native_phase_controls_v1';p=read(d/'protocol.json');done=read(d/'complete.json')
 assert done['complete']and read(d/'independent_audit.json')['passed']
 states={}
 for r in done['controls']:
  assert digest(r['file'])==r['sha256'];states[r['mode']]=load_packet(r['file'])
 baseline=FrozenPredictor();states['correct']=baseline.state
 return baseline,states
def collect(records):
 rows=[];arrays=[];sourcefiles={};case_status=[];sham_checks=0
 for record in records:
  folder=Path(record['folder']);case=record['case_id']
  if not record.get('available',True):
   case_status.append(dict(case_id=case,available=False,reason=record.get('reason','native_no_legal_event')));continue
  history=load_packet(folder/'observed_history.npz');source_query=len(history['queries'])-1
  zmeta=read(folder/'Z_reference_live_feedback.json');smeta=read(folder/'Z_sham_live_feedback.json')
  assert zmeta['rows']==smeta['rows']and [(q['query'],q['tick'],q['rows'])for q in zmeta['queries']]==[(q['query'],q['tick'],q['rows'])for q in smeta['queries']];sham_checks+=1
  if zmeta['rows']:
   za=load_packet(folder/'Z_reference_live_local.npz');sa=load_packet(folder/'Z_sham_live_local.npz')
   assert set(za)==set(sa)
   for k in za:assert np.array_equal(za[k],sa[k],equal_nan=True);sham_checks+=1
  case_status.append(dict(case_id=case,available=True,known_initial_identity=zmeta['owner']['known_earliest_identity'],Z_local_rows=len(zmeta['rows']),source_event_query=source_query))
  # Z_sham is a software control, never a second copy of Z in primary metrics.
  for branch in('Z_reference','C_original'):
   livefile=folder/(branch+'_live_feedback.json');live=read(livefile);sourcefiles[str(livefile)]=digest(livefile)
   if not live['rows']:continue
   file=folder/(branch+'_live_local.npz');a=load_packet(file);sourcefiles[str(file)]=digest(file)
   receiptfile=folder/(branch+'_origin_receipt.json');receipt=read(receiptfile);sourcefiles[str(receiptfile)]=digest(receiptfile)
   origin=load_packet(folder/(branch+'_origin.npz'));originmeta=read(folder/(branch+'_origin.json'))
   origin_tick=int(originmeta['absolute_start_tick']);closed=receipt['completion']
   tick=closed[0].get('requested95_tick')if closed else None
   ready=bool(receipt['feedback_ready']and receipt['commands_returned']==32 and tick is not None)
   phase=float(origin['dt_s'][:tick].sum())if ready else None
   clock={origin_tick:0.};offset=0.
   for name in('origin','following','delayed'):
    file=folder/(branch+'_'+name+'.npz')
    if not file.exists():continue
    packet=load_packet(file);meta=read(folder/(branch+'_'+name+'.json'));at=int(meta['absolute_start_tick']);elapsed=np.r_[0.,np.cumsum(packet['dt_s'])]
    for i in {r['start_tick']for r in live['rows']}|{r['end_tick']for r in live['rows']}:
     if at<=i<=at+len(packet['dt_s']):clock[i]=offset+float(elapsed[i-at])
    offset+=float(packet['dt_s'].sum())
   allocation={}
   for q in live['queries']:
    for i in q['rows']:
     assert i not in allocation and live['rows'][i]['end_tick']<=q['tick'];allocation[i]=q['query']
     assert source_query<q['query']<=source_query+2
   for i,r in enumerate(live['rows']):
    assert r['start_tick']in clock and r['end_tick']in clock
    if ready:
     e=qualify(start_s=clock[r['start_tick']],end_s=clock[r['end_tick']],phase95_s=phase,
      original_ready=True,original_receipt_tick=origin_tick+len(origin['dt_s']),end_tick=r['end_tick'])
    else:e=dict(available=False,reason='original32_or95_incomplete')
    primary_horizon=r['command_query']<=source_query+1
    if i in allocation:assert e['available']and primary_horizon
    row=dict(case_id=case,scene_id=case,task=case.split(':')[0],seed=int(case.split(':')[1]),branch=branch,
     object_index=r['object_index'],start_tick=r['start_tick'],end_tick=r['end_tick'],dt_s=r['dt_s'],
     source_event_query=source_query,actual_command_query=r['command_query'],
     stage_eligible=bool(e['available']),primary_horizon=primary_horizon,
     eligibility_reason=e['reason'],packet_query=allocation.get(i),local_row=i,live_file=str(livefile))
    rows.append(row);arrays.append({k:a[k][i]for k in a})
 if not rows:return rows,{},sourcefiles,case_status,sham_checks
 return rows,{k:np.stack([r[k]for r in arrays])for k in arrays[0]},sourcefiles,case_status,sham_checks
def wrong_commands(x,a,meta):
 study=module('frozen_old_phase_donor_population',TTA/'native_phase_feedback_v1/study.py')
 _,oldx,olda,oldmeta,eligible,_=study.dataset();train=np.flatnonzero(eligible);assert len(train)==3164
 result=a['command'].copy();donors=[]
 for i,r in enumerate(meta):
  choices=np.array([j for j in train if oldmeta[j]['seed']!=r['seed']and oldmeta[j]['task']==r['task']and oldmeta[j]['object_index']==r['object_index']],int)
  assert len(choices);distance=np.abs(np.array([oldmeta[j]['dt_s']for j in choices])-r['dt_s']);near=choices[distance==distance.min()]
  score=np.mean((oldx[near]-x[i])**2,axis=1);j=int(near[np.argmin(score)])
  result[i]=olda['command'][j];result[i,38]=a['command'][i,38];donors.append(j)
 return result,donors
def evaluate(records,*,independent,expected_cases=32):
 predictor,states=controls();meta,a,sourcefiles,case_status,sham_checks=collect(records)
 if not meta:return dict(complete=True,independent_feedback_gate_passed=False,reason='no_actual_local_rows',requested=expected_cases,case_status=case_status),{},sourcefiles
 x=a['context'];wrong,donors=wrong_commands(x,a,meta);pred={}
 for mode in MODES:
  xx,v,k,_=predictor.model.coordinates(x,wrong if mode=='wrong_command'else a['command'],a['start_points'],mode)
  pred[mode]=predictor.model.predict(states[mode],xx,v,k)
  assert np.isfinite(pred[mode]).all()
 # The correct kernel was computed before actual query draw by the online shadow path.
 by_stream={}
 for i,r in enumerate(meta):by_stream.setdefault(r['live_file'],[]).append(i)
 deployed_checks=0
 for file,ids in by_stream.items():
  live=read(file);lookup={meta[i]['local_row']:i for i in ids}
  for q in live['queries']:
   rr=[lookup[i]for i in q['rows']]
   actual=predictor.predict([dict(context=x[i],command=a['command'][i],start_points=a['start_points'][i])for i in rr])
   assert np.array_equal(actual,np.asarray(q['prediction'],float).reshape(-1,2));deployed_checks+=len(rr)
   # Primary packet scoring uses the exact deployed batching of this checkpoint.
   pred['correct'][rr]=actual
 metrics=module('fixed_independent_duration_whole_seed_metrics',TTA/'silhouette_geometry_state_v1/metrics.py')
 comparison=module('fixed_independent_paired_control_metrics',TTA/'native_origin_response_v1/study.py').compare
 def score(ids):
  if not len(ids):return dict(available=False,rows=0)
  yy=a['target'][ids];mm=[meta[i]for i in ids]
  return dict(available=True,rows=len(ids),scores={mode:metrics.summary(p[ids],yy,mm)for mode,p in pred.items()},
    paired_controls={mode:comparison(pred['correct'][ids],pred[mode][ids],yy,mm)for mode in MODES[1:]})
 phase=np.array([r['stage_eligible']and r['primary_horizon']for r in meta]);packet=np.array([r['packet_query']is not None for r in meta])
 strata=dict(all_postphase=np.flatnonzero(phase),single_use_packet_union=np.flatnonzero(packet),
  actual_q1=np.array([i for i,r in enumerate(meta)if r['packet_query']==r['source_event_query']+1],int),
  actual_q2=np.array([i for i,r in enumerate(meta)if r['packet_query']==r['source_event_query']+2],int),
  retained_prephase_or_shortlag=np.flatnonzero(~np.array([r['stage_eligible']for r in meta])),
  retained_after_q2_no_later_query=np.flatnonzero(~np.array([r['primary_horizon']for r in meta])))
 results={name:score(ids)for name,ids in strata.items()};passing=True;reasons=[]
 def positive(v,confidence=True):
  if not v.get('available'):return False
  s=v['scores']['correct'];return all(s['summary'].get(t,{}).get('improvement',-1)>0 for t in TASKS)and(not confidence or s['whole_seed_bootstrap95'][0]>0)
 for name in('all_postphase','single_use_packet_union'):
  v=results[name];ok=positive(v)
  if v.get('available'):ok&=all(v['paired_controls'][m]['whole_seed_bootstrap95'][0]>0 for m in('wrong_command','history_only'))
  passing&=ok
  if not ok:reasons.append(name+'_primary_or_control_gate')
 for name in('actual_q1','actual_q2'):
  ok=positive(results[name],False);passing&=ok
  if not ok:reasons.append(name+'_both_tasks_positive')
 lookup={(r['case_id'],r['branch'],r['start_tick'],r['end_tick'],r['object_index']):i for i,r in enumerate(meta)}
 ids=[];mate=[];membership=set(strata['single_use_packet_union'].tolist())
 for i in membership:
  r=meta[i];key=(r['case_id'],r['branch'],r['start_tick'],r['end_tick'],1)
  if r['object_index']==0 and bool(a['relative_available'][i])and key in lookup and lookup[key]in membership:ids.append(i);mate.append(lookup[key])
 relative=None
 if ids:
  relative=metrics.summary(pred['correct'][ids]-pred['correct'][mate],a['relative_target'][ids],[dict(meta[i],object_index=-1)for i in ids])
  ok=all(relative['summary'].get(t,{}).get('improvement',-1)>0 for t in TASKS)and relative['whole_seed_bootstrap95'][0]>0
 else:ok=False
 passing&=ok
 if not ok:reasons.append('relative_packet_gate')
 coverage={t:len({r['seed']for i,r in enumerate(meta)if r['task']==t and packet[i]})for t in TASKS}
 enough=all(n>=8 for n in coverage.values());passing&=enough
 if not enough:reasons.append('minimum_eight_packet_seed_groups_per_task')
 report=dict(complete=True,independent_feedback_gate_passed=bool(passing and independent),
  scope='independent_TRAIN_confirmation'if independent else'old_developer_contract_replay_NOT_independent',
  gate_numeric_support_only=bool(passing),failed_gates=reasons,requested=expected_cases,case_status=case_status,
  all_rows=len(meta),results=results,relative_packet=relative,relative_windows=len(ids),packet_seed_coverage=coverage,
  prequery_prediction_exact_checks=deployed_checks,Z_sham_control_checks=sham_checks,
  post_q2_rows_retained_but_not_used_in_original_horizon_primary=True,metadata=meta,old_FIT_donor_indices=donors,
  no_new_fit_scaling_or_model_selection=True,model_fit_steps=0,online_updates=0,TTA_utility_proven=False)
 return report,dict(**pred,target=a['target'],phase_eligible=phase,packet_eligible=packet),sourcefiles
def analyze(root):
 root=Path(root);spec=read(OUTPUT/'protocol.json');verify(spec);assert not spec['cpu_fixture']
 result=read(root/'result.json');assert result['complete'];manifest=read(root/'manifest.json');j=load_journal(root,manifest,spec)
 assert len(j['cases'])==len(spec['scenes'])and all(read(root/r['folder']/'live_feedback_audit.json')['passed']for r in j['cases']if r['audit']['available'])
 records=[dict(case_id=r['case_id'],folder=str(root/r['folder']),available=r['audit']['available'],reason=r['audit'].get('reason'))for r in j['cases'][2:]]
 report,arrays,inputs=evaluate(records,independent=True,expected_cases=32)
 attempts=list((root/'host_attempts').iterdir());qualified=[]
 for attempt in attempts:
  exitfile=attempt/'barrier/normal_child_exit.json';resource=attempt/'resources.jsonl'
  if not exitfile.exists()or not resource.exists():continue
  child=read(exitfile);samples=[json.loads(s)for s in resource.read_text().splitlines()]
  if(child['normal_exit']and child['exit_code']==0 and child['proc_absent_after_wait']
      and child['protocol_identity']==spec['identity']and samples[-1].get('own_gpu_released')
      and not any('stopped'in r for r in samples)):
   long_samples=[r for r in samples if r.get('seconds',0)>=300 and r.get('gpu_pids')]
   if long_samples:qualified.append(str(attempt))
 resource_ok=len(qualified)==1 and not(root/'resource_blocked.json').exists()
 report['independent_feedback_gate_passed']&=resource_ok
 if not resource_ok:report.setdefault('failed_gates',[]).append('long_runtime_resource_or_normal_exit_qualification')
 recovery_file=TTA/'outputs/native_feedback_recovery_v2/whole_case_audit.json';recovery=read(recovery_file)
 assert recovery['passed']and recovery['whole_cases_verified']==20
 report.update(complete_whole_cases_reused=20,new_GPU_continuation_cases=len(spec['scenes'])-20,
  prior_source_actual_host_exit=1,prior_source_resource_qualification_passed=False,
  prior_source_whole_case_CPU_audit=str(recovery_file),prior_source_whole_case_CPU_audit_sha256=digest(recovery_file),
  resource_qualification_scope='This new12 GPU0 four-lane continuation under user-authorized34 floor only; all prior source host/resource failures remain unchanged. All complete data reused exhaustively after full CPU audit, without partial splicing.')
 report.update(resource_qualification_verified=resource_ok,qualified_actual_attempts=qualified,
  actual_host_return0_finalization_still_required=True,
  candidate_identity=spec['candidate_identity'],protocol_identity=spec['identity'],
  initial_missing_cases=spec['initial_missing_cases'],independent_requested_cases=32,
  journal_sha256=digest(root/'journal.json'),input_sha256=inputs,direction_training_allowed=False,eligible_for_direction_training_after_host0=report['independent_feedback_gate_passed'])
 with(root/'fixed_feedback_predictions.npz').open('xb')as f:np.savez_compressed(f,**arrays)
 report['predictions_sha256']=digest(root/'fixed_feedback_predictions.npz')
 write(root/'independent_feedback_gate.json',report)
 print('INDEPENDENT_FIXED_FEEDBACK_GATE',report['independent_feedback_gate_passed'],report.get('failed_gates'),report.get('packet_seed_coverage'),flush=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();analyze(a.dataset)
