"""Actual released full packets and legal post-feedback opportunities, no labels."""
from pathlib import Path
import hashlib,json,numpy as np
from .common import read,sha,mapped
from native_feedback_after_train_v1.targets import first_action

def arrays(p):
 with np.load(mapped(p),allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def index(folder,config):
 folder=Path(folder);result=read(folder/'result.json');assert result['complete']and result['condition']=='M_event_full'and result['native_episode_complete']
 queries=result['queries'];sources=[read(f)for f in folder.glob('source*/source.json')];packets=[];times=[0.]
 for r in queries:
  trace=arrays(r['execution']['actual_trace_file']);times.extend((np.asarray(trace['dt_s']).cumsum()+times[-1]).tolist())
 for row in queries:
  count=row['response'].get('released_packet_rows',0)
  if not count:continue
  q=row['query'];matches=[s for s in sources if s['feedback_consumed']and s['retired_at_query']==q];assert len(matches)==1
  s=matches[0];source=s['source_query'];assert q in(source+1,source+2)
  boundary=next(b for b in s['boundaries']if b['query']==q);ids=boundary['eligible_rows'];assert len(ids)==count and ids
  assert sha(mapped(s['local_file']))==s['local_sha256'];local=arrays(s['local_file']);origin=read(folder/('execution%03d.json'%source));assert origin['complete_source_chunk']and len(origin['receipts'])==32 and all(x['status']=='returned'for x in origin['receipts'])
  phase=origin['completion'][0]['requested95_tick'];assert phase is not None;phase_global=queries[source]['tick']+phase
  for i in ids:
   assert local['start_tick'][i]>=phase_global and local['end_tick'][i]<=row['tick']and times[int(local['end_tick'][i])]-times[phase_global]>=.25-1e-9
   assert local['command_query'][i]in(source,source+1)and local['command_query'][i]<q
  proof=dict(query=q,tick=row['tick'],observation_sha256=row['observation']['sha256'],policy_sha256=row['policy']['policy_sha256']);digest=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
  packets.append(dict(source_query=source,query=q,tick=row['tick'],row_ids=ids,query_proof_sha256=digest,single_use=True,within_original_q1_q2_horizon=True,source_local_file=s['local_file'],source_local_sha256=s['local_sha256']))
 views=[dict(r,target=dict(valid_command_count=r['execution']['commands_returned']))for r in queries];actions=[]
 for packet in packets:
  choice=first_action(views,packet,config,other_packets=packets)
  if choice is not None:actions.append(dict(packet=packet,action_query=choice['query'],application_mode=choice['mode'],actual_teacher_label_available=False))
 return dict(packets=packets,potential_actions=actions,labels_available=False,native_full_complete=True,no_policy_outcome_filter=True)
