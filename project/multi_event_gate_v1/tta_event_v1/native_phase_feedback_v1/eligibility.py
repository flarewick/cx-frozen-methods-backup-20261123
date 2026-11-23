"""Actual phase-qualified loss windows; future boundary never silently borrowed."""
from pathlib import Path
import sys,numpy as np
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_phase_feedback_v1'
sys.path.insert(0,str(T/'native_arrived_feedback_v1'))
from review import read,load,sha,write
MIN_LAG_S=.25
def qualify(*,start_s,end_s,phase95_s,original_ready,original_receipt_tick,end_tick):
 if not all(np.isfinite(v)for v in(start_s,end_s,phase95_s))or not start_s<end_s:raise ValueError('Actual monotonic physical clock required')
 if not original_ready:return dict(available=False,reason='original32_or95_incomplete',available_tick=None)
 if start_s<phase95_s:return dict(available=False,reason='window_starts_before_actual95',available_tick=None)
 if end_s-phase95_s<MIN_LAG_S:return dict(available=False,reason='response_minlag0p25_not_arrived',available_tick=None)
 return dict(available=True,reason='actual_post95_window_and_response_lag_arrived',available_tick=max(original_receipt_tick,end_tick))
def run():
 local=T/'outputs/native_arrived_feedback_v1/local';p=read(local/'protocol.json');meta=p['metadata'];assert read(local/'independent_audit.json')['passed']
 spec=read(T/'outputs/native_arrived_mask_v3/protocol.json');jobs={j['index']:j for j in spec['jobs']};cache={};rows=[];counts={};checks=0
 for i,r in enumerate(meta):
  j=jobs[r['stream_index']]
  if j['index']not in cache:
   origin=j['stream']['segments'][0];trace=load(origin['actuation']);tick=origin['previous_origin_requested95_absolute_tick']
   assert tick is not None and j['origin_feedback_ready'];checks+=1
   rel=tick-origin['absolute_start_tick'];assert 0<rel<=len(trace['dt_s']);checks+=1
   phase=float(np.sum(trace['dt_s'][:rel]));frames={f['global_tick']:f['elapsed_selected_s']for f in j['stream']['frames']}
   cache[j['index']]=(origin,phase,frames)
  origin,phase,frames=cache[j['index']];start=frames[r['start_tick']];end=frames[r['end_tick']]
  assert abs(end-start-r['dt_s'])<1e-9;checks+=1
  result=qualify(start_s=start,end_s=end,phase95_s=phase,original_ready=j['origin_feedback_ready'],
    original_receipt_tick=origin['annotation_available_tick'],end_tick=r['end_tick'])
  counts[result['reason']]=counts.get(result['reason'],0)+1
  earliest=None;status='not_eligible'
  if result['available']:
   if result['available_tick']<=origin['annotation_available_tick']and j['stream']['following_query_sampled']:
    earliest=origin['annotation_available_tick'];status='actual_recorded_following_query_boundary'
   else:
    following=[s for s in j['stream']['segments']if s['kind']=='following']
    if following and following[0]['complete_native32']:
     earliest=following[0]['annotation_available_tick'];status='would_require_NOT_collected_third_query'
    else:status='no_full_following_or_nextquery_censored'
  rows.append(dict(row=i,case_id=r['case_id'],branch=r['branch'],segment=r['segment'],start_tick=r['start_tick'],end_tick=r['end_tick'],
    start_selected_s=start,end_selected_s=end,actual95_selected_s=phase,
    measurement=result,hypothetical_earliest_query_tick=earliest,boundary_status=status))
 report=dict(complete=True,checks=checks,all_rows=len(rows),rows=rows,counts=counts,original_requested_cases=32,missing_cases=p['missing_cases'],
  min_response_lag_s=MIN_LAG_S,source_sha256=sha(Path(__file__)),local_protocol_sha256=sha(local/'protocol.json'),
  previous_failed_gate_preserved=True,actual_third_query_collected=False,online_updates=0,full_candidate_frozen=False,TTA_utility_proven=False)
 write(O/'eligibility.json',report)
 print('ACTUAL_PHASE_WINDOW_ELIGIBILITY',counts,{k:sum(r['boundary_status']==k for r in rows)for k in('actual_recorded_following_query_boundary','would_require_NOT_collected_third_query','no_full_following_or_nextquery_censored')},flush=True)
if __name__=='__main__':run()
