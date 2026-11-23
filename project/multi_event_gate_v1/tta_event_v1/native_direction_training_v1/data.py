"""Strict same-state, issuing-query and valid-command binding for offline meta examples."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
T=Path(__file__).resolve().parents[1]
ARM=np.asarray([0,1,2,3,4,5,7,8,9,10,11,12],np.int64)
TASKS=('stack_blocks_two','stack_bowls_two')
CANDIDATE='159fc7b7c49a200527c557fc9c7df9460a0b8dff8b836b5e2fa67f7f95ebe23e'
KERNEL='86069427fa4a4e2f9386227e35bdaebbcd18a70773dff4b7686ef83b854fb0c1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def tensor(a):return torch.as_tensor(np.array(a,copy=True),dtype=torch.float32)
class PacketReader:
 def __init__(self):self.files={};self.local_validated=set();self.observed_hashes={}
 def file(self,path,digest):
  path=str(path)
  if path not in self.files:
   assert sha(path)==digest,('changed_input',path);self.files[path]=load(path);self.observed_hashes[path]=digest
  else:assert self.observed_hashes[path]==digest
  return self.files[path]
 def packet(self,p,queries):
  source=p['source_query'];q=p['query'];assert q in(source+1,source+2)and p['single_use']and p['offline_training_packet']
  byq={r['query']:r for r in queries};assert len(byq)==len(queries)and sorted(byq)==list(range(len(queries)))
  origin=byq[source];nextq=byq[q];assert p['tick']==nextq['tick']and p['next_policy_file']==nextq['policy']['policy_file']and p['next_policy_sha256']==nextq['policy']['policy_sha256']
  trace=nextq['trace'];a=self.file(trace['file'],trace['sha256']);labels=self.file(p['teacher_target_file'],p['teacher_target_sha256']);policy=self.file(p['next_policy_file'],p['next_policy_sha256'])
  assert nextq['target']['file']==p['teacher_target_file']and nextq['target']['sha256']==p['teacher_target_sha256']
  assert nextq['policy']['actor_observation_tick']==p['tick']and nextq['policy']['observation_sha256']==nextq['observation']['sha256']
  obs=self.file(nextq['observation']['file'],nextq['observation']['sha256'])
  actor_obs=self.file(nextq['policy']['observation_file'],nextq['policy']['observation_sha256'])
  assert set(actor_obs)==set(obs)and all(np.array_equal(actor_obs[k],obs[k])for k in obs),'Same tick and byte-exact actual observation required'
  valid=np.asarray([r['offset']for r in trace['receipts']if r['status']=='returned'],np.int64)
  assert labels['valid_offsets'].dtype==np.int64 and np.array_equal(valid,labels['valid_offsets'])and len(valid)==p['valid_next_commands']>0
  assert len(set(valid.tolist()))==len(valid)and np.all((valid>=0)&(valid<32))and np.max(valid)<len(a['source_commands'])
  assert np.array_equal(labels['teacher_ARM'],a['source_commands'][valid][:,ARM])and np.array_equal(labels['once_base_ARM'],policy['base'][valid][:,ARM])
  assert np.array_equal(labels['direction_target_ARM'],np.clip(labels['teacher_ARM']-labels['once_base_ARM'],-.1,.1).astype(np.float32))
  origin_trace=origin['trace'];assert origin_trace['complete_source_chunk']and len(origin_trace['receipts'])==32 and all(r['status']=='returned'for r in origin_trace['receipts'])
  assert len(origin_trace['completion'])==1;phase=origin_trace['completion'][0];assert phase['status']=='feedback_ready'and phase['requested95_tick']is not None
  phase_tick=origin['tick']+phase['requested95_tick'];actual_origin=self.file(origin_trace['file'],origin_trace['sha256'])
  phase_s=float(np.sum(actual_origin['dt_s'][:phase['requested95_tick']]));elapsed_s=0.
  for i in range(source,q):
   segment=byq[i]['trace'];dt=self.file(segment['file'],segment['sha256'])['dt_s'];elapsed_s+=float(np.sum(dt))
  assert elapsed_s>=phase_s+.25
  local=self.file(p['local_file'],p['local_sha256']);ids=np.asarray(p['row_ids'],np.int64)
  assert len(ids)>0 and len(set(ids.tolist()))==len(ids)and ids.min()>=0 and ids.max()<len(local['target'])
  assert np.all(local['start_tick'][ids]>=phase_tick)and np.all(local['end_tick'][ids]<=p['tick'])
  issuing=local['command_query'][ids];assert set(issuing.tolist())<=set((source,source+1))and sorted(set(issuing.tolist()))==p['actual_issuing_queries']
  for commandq in set(issuing.tolist()):
   record=byq[commandq]['policy'];original=self.file(record['policy_file'],record['policy_sha256'])
   assert np.array_equal(local['issuing_query_hidden'][ids[issuing==commandq]],np.repeat(original['hidden'][None],np.sum(issuing==commandq),axis=0))
  for row_id,commandq in zip(ids,issuing,strict=True):
   segment=byq[int(commandq)]['trace'];segment_dt=self.file(segment['file'],segment['sha256'])['dt_s']
   before_segment=sum(float(np.sum(self.file(byq[i]['trace']['file'],byq[i]['trace']['sha256'])['dt_s']))for i in range(source,int(commandq)))
   prefix=int(local['end_tick'][row_id])-segment['global_start_tick'];assert 0<prefix<=len(segment_dt)
   assert before_segment+float(np.sum(segment_dt[:prefix]))>=phase_s+.25
  if p['local_file']not in self.local_validated:
   from native_delayed_query_v1.predictor import FrozenPredictor
   predictor=FrozenPredictor();xx,u,kind,_=predictor.model.coordinates(local['context'],local['command'],local['start_points'],'correct')
   predicted=predictor.predict([dict(context=local['context'][i],command=local['command'][i],start_points=local['start_points'][i])for i in range(len(local['target']))])
   assert np.array_equal(local['model_context'],xx)and np.array_equal(local['model_command'],u)and np.array_equal(local['model_kind'],kind)and np.array_equal(local['kernel_prediction'],predicted)
   self.local_validated.add(p['local_file'])
  arrays=dict(hidden=local['issuing_query_hidden'][ids],context=local['model_context'][ids],command=local['model_command'][ids],kind=local['model_kind'][ids],kernel_prediction=local['kernel_prediction'][ids],target=local['target'][ids],
   next_hidden=policy['hidden'][None],next_logits=policy['logits'][:,ARM][None],next_reference=policy['raw_delta'][:,ARM][None],teacher_target=labels['direction_target_ARM'])
  assert all(np.isfinite(v).all()for v in arrays.values())and arrays['hidden'].shape==(len(ids),32,64)and arrays['teacher_target'].shape==(len(valid),12)
  result={k:tensor(v)for k,v in arrays.items()};result['valid_offsets']=torch.from_numpy(valid.copy())
  return result

def full_manifest(path):
 path=Path(path).resolve();assert path.is_relative_to((T/'outputs').resolve())
 d=read(path)
 assert d['complete']and d['full_original_expert_cache_ready']and d['all_actual_issuing_query_hidden_bound']and d['actual_phase95_plus025_and_single_use_verified']
 assert d['candidate_identity']==CANDIDATE and d['original80FIT20VAL_split_unchanged']and d['all_original_cases_processed_without_outcome_filter']
 originalfile=T/'outputs/native_expert_full_inputs_v1/manifest.json';original=read(originalfile)
 assert d['original100_manifest']==str(originalfile)and d['original100_manifest_sha256']==sha(originalfile)
 expected={r['case_id']:r for r in original['records']};cases={r['case_id']:r for r in d['cases']};assert len(cases)==len(d['cases'])==len(expected)==100 and set(cases)==set(expected)
 counts={};packets={task:{split:[]for split in('adapter_fit','adapter_validation')}for task in TASKS};used=set()
 for key,c in cases.items():
  r=expected[key];assert c['original_expert_index']==r['index']and c['split']==r['split']and c['task']==r['task']and c['native_available']and c['feedback_processing_complete']
  k=c['task']+'/'+c['split'];counts[k]=counts.get(k,0)+1
 assert counts==original['counts']and sorted(counts.values())==[10,10,40,40]
 for p in d['packets']:
  c=cases[p['case_id']];assert p['original_expert_index']==c['original_expert_index']and p['split']==c['split']
  key=(p['case_id'],p['source_query']);assert key not in used;used.add(key);packets[c['task']][p['split']].append(p)
 for c in cases.values():assert sum(p['case_id']==c['case_id']for p in d['packets'])<=3
 for task in TASKS:
  assert len(packets[task]['adapter_fit'])>=8 and packets[task]['adapter_validation'],'Both full original split populations must have supported feedback'
  for group in packets[task].values():group.sort(key=lambda p:(p['original_expert_index'],p['source_query'],p['query']))
 for group in('source_sha256','input_sha256'):
  for p,h in d[group].items():assert sha(p)==h,('full_cache_source_changed',p)
 return d,cases,packets
