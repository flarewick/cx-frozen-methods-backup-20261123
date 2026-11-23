"""Keep current arrived-packet inputs; add explicit OFFLINE next selected-action labels."""
from pathlib import Path
import numpy as np
from native_direction_training_v2.data import T,ARM,read,sha,PacketReader,full_manifest

def inventory():
 manifest_file=T/'outputs/native_direction_cache_contract_v2/qualified_manifest.json'
 manifest,cases,pools=full_manifest(manifest_file)
 file=T/'outputs/native_outer_alignment_audit_v1/report.json';report=read(file);protocol=read(file.parent/'protocol.json')
 assert report['passed']and report['complete']and report['identities_retained']==100 and report['packets_retained']==261 and report['not_training_authorization']
 assert protocol['qualified_manifest_sha256']==sha(manifest_file)
 for p,h in protocol['source_sha256'].items():assert sha(p)==h
 records={(r['case_id'],r['source_query'],r['feedback_query']):r for r in report['packets']};assert len(records)==261
 plans=[];excluded=[]
 for p in manifest['packets']:
  row=records[p['case_id'],p['source_query'],p['query']];assert row['source_would_be_selected']
  nextq=row['next_selected_at_or_after_feedback'];record=dict(packet=p,alignment=row)
  if nextq is None or not nextq['valid_command_count']:
   excluded.append(dict(case_id=p['case_id'],source_query=p['source_query'],query=p['query'],split=p['split'],reason='no_remaining_selected_action'if nextq is None else'selected_action_has_no_returned_teacher_commands'))
  else:plans.append(record)
 assert len(plans)==174 and len(excluded)==87 and len(cases)==100
 return manifest,cases,plans,excluded

class SelectedReader(PacketReader):
 def packet_for_action(self,p,queries,selected_query):
  if selected_query is None:raise ValueError('No remaining selected action; retain packet without outer loss')
  if type(selected_query)is not int or selected_query<p['query']:raise ValueError('Outer action target cannot precede feedback arrival')
  value=self.packet(p,queries);byq={q['query']:q for q in queries};future=byq[selected_query]
  target=future['target'];policy=future['policy'];trace=future['trace']
  assert future['tick']>=p['tick']and policy['actor_observation_tick']==future['tick']
  labels=self.file(target['file'],target['sha256']);pred=self.file(policy['policy_file'],policy['policy_sha256']);actual=self.file(trace['file'],trace['sha256'])
  offsets=np.asarray([r['offset']for r in trace['receipts']if r['status']=='returned'],np.int64)
  assert len(offsets)>0 and np.array_equal(offsets,labels['valid_offsets'])and len(offsets)==target['valid_command_count']
  assert np.all((offsets>=0)&(offsets<32))and np.max(offsets)<len(actual['source_commands'])
  assert np.array_equal(labels['teacher_ARM'],actual['source_commands'][offsets][:,ARM])and np.array_equal(labels['once_base_ARM'],pred['base'][offsets][:,ARM])
  assert np.array_equal(labels['direction_target_ARM'],np.clip(labels['teacher_ARM']-labels['once_base_ARM'],-.1,.1).astype(np.float32))
  obs=self.file(future['observation']['file'],future['observation']['sha256']);actor_obs=self.file(policy['observation_file'],policy['observation_sha256'])
  assert obs.keys()==actor_obs.keys()and all(np.array_equal(obs[k],actor_obs[k])for k in obs)
  assert policy['observation_sha256']==future['observation']['sha256']and target['actor_observation_tick']==future['tick']and target['observation_sha256']==future['observation']['sha256']
  from native_direction_training_v1.data import tensor
  added=dict(action_hidden=pred['hidden'][None],action_logits=pred['logits'][:,ARM][None],action_reference=pred['raw_delta'][:,ARM][None],action_teacher_target=labels['direction_target_ARM'])
  assert all(np.isfinite(a).all()for a in added.values())
  value.update({k:tensor(v)for k,v in added.items()});value['action_valid_offsets']=__import__('torch').from_numpy(offsets.copy())
  value['offline_action_query']=selected_query;value['actual_feedback_query']=p['query']
  return value
