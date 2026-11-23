"""Replay actual52 RGB streams and ONLY previously recorded q+1 query boundaries."""
from pathlib import Path
import hashlib,sys,numpy as np
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_phase_feedback_v1'
from feedback_queue import FeedbackQueue
sys.path.insert(0,str(T/'native_arrived_mask_v3'))
from data import LoadedStream,read,sha,load
sys.path.insert(0,str(T/'native_arrived_feedback_v1'));from review import write
def image_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def run():
 spec=read(T/'outputs/native_arrived_mask_v3/protocol.json');local=read(T/'outputs/native_arrived_feedback_v1/local/protocol.json')
 phase=read(O/'eligibility.json');combined=read(T/'outputs/native_current_v4/collections/de8b2c74c8d40f23b7177c9d/combined_current_source.json')
 sources={r['case_id']:r for r in combined['rows']};checks=0;rows=[];packet_rows=[];inputs={}
 for job in spec['jobs']:
  data=LoadedStream(job);origin=job['stream']['segments'][0];frames=job['stream']['frames'];lookup={f['global_tick']:f for f in frames}
  grouped={}
  for i,r in enumerate(local['metadata']):
   if r['stream_index']==job['index']:grouped.setdefault(r['end_tick'],[]).append((i,r))
  q=FeedbackQueue();q.arrive(job['selected_origin_tick'],0.,image_sha(data.history['cam0'][-1]))
  ticket=q.register(len(data.history['queries'])-1);query_done=False;emitted=[];ignored=0
  trace=load(origin['actuation']);pt=origin['previous_origin_requested95_absolute_tick']
  phase_time=None if pt is None else float(np.sum(trace['dt_s'][:pt-origin['absolute_start_tick']]))
  for images,meta in data.native():
   tick=meta['global_tick'];q.arrive(tick,lookup[tick]['elapsed_selected_s'],image_sha(images[0]))
   for i,r in grouped.get(tick,[]):
    seg=next(s for s in job['stream']['segments']if s['kind']==r['segment'])
    okay=q.window(ticket,row=i,start_tick=r['start_tick'],end_tick=r['end_tick'],kind=r['segment'],
     command_start_tick=seg['absolute_start_tick'],command_end_tick=tick)
    ignored+=not okay;checks+=1
   if tick==origin['annotation_available_tick']:
    q.close_origin(ticket,commands_returned=origin['commands_returned'],complete_native32=origin['complete_native32'],
     receipt_tick=tick,phase95_tick=pt,phase95_elapsed_s=phase_time)
    if job['stream']['following_query_sampled']:
     file=Path(job['history']).parent/(job['branch']+'_following_query.json');query=read(file);digest=sha(file)
     # The native source record already contains this exact immutable query evidence.
     rel=str(file.relative_to(Path(job['source_root'])))
     assert sources[job['case_id']]['source_record']['evidence_sha256'][rel]==digest;checks+=1
     assert query['actor_observation_tick']==tick and query['query']==len(data.history['queries']);checks+=1
     inputs[str(file)]=digest
     packet=q.at_query(query=query['query'],tick=tick,query_proof_sha256=digest);query_done=True
     expected=tuple(r['row']for r in phase['rows']if r['row']in{i for vals in grouped.values()for i,_ in vals}
       and r['measurement']['available']and r['measurement']['available_tick']<=tick)
     assert(packet.rows if packet is not None else())==expected;checks+=1
     if packet is not None:emitted.append(packet);packet_rows.extend(packet.rows)
  pending=q.eligible(ticket);done=q.finish()
  assert len(emitted)<=1 and done['released_feedback_packets']==len(emitted);checks+=1
  rows.append(dict(case_id=job['case_id'],branch=job['branch'],actual_q1_boundary_replayed=query_done,
   released_CPU_feedback_packets=len(emitted),released_observation_rows=sum(len(p.rows)for p in emitted),
   source_consumed_once=bool(emitted),later_windows_ignored_for_consumed_source=ignored,
   remaining_postphase_rows_without_recorded_nextquery=len(pending),episode_end_pending_sources=done['pending_sources']))
  print('ACTUAL_PHASE_QUERY_BOUNDARY_REPLAY',job['index'],len(emitted),len(pending),flush=True)
 report=dict(passed=True,checks=checks,streams=len(rows),rows=rows,
  CPU_replayed_feedback_packets=sum(r['released_CPU_feedback_packets']for r in rows),
  CPU_released_observation_rows=len(packet_rows),released_row_indices=packet_rows,
  original_requested_cases=32,missing_cases=spec['missing'],source_sha256={str(p):sha(p)for p in(C/'replay.py',C/'feedback_queue.py',C/'eligibility.py')},
  input_sha256=inputs,actual_recorded_third_query_boundaries=0,
  scope='CPU replay of immutable actual arrived RGB and existing query evidence only. No native queue/optimizer/actor integration; no hypothetical third query called.',
  old_failed_response_gates_preserved=True,online_updates=0,TTA_utility_proven=False)
 write(O/'actual_query_replay.json',report);print('ACTUAL52_RECORDED_QUERY_FEEDBACK_REPLAY_PASS',checks,report['CPU_replayed_feedback_packets'],len(packet_rows),flush=True)
if __name__=='__main__':run()
