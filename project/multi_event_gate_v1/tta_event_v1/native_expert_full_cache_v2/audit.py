"""CPU reconstruction of actual teacher drives, passive frames, once-base and head outputs."""
from pathlib import Path
import numpy as np
import torch
from .paths import read,write,sha,verify,O
from cx_tta.structured_features import pack_observable
from cx_tta.structured_action_adapter import FORWARD_KEYS,ARM_INDICES
from cx_tta.eval_diagnostics import array_hash
from native_nextchunk_v1.sampler_fork import base_hash
from native_stage_v1.clock import COMMON_TICKS
ARM=np.asarray(ARM_INDICES,np.int64)
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def policy_check(row,obs,prior,head,robots,projection):
 assert sha(row['policy_file'])==row['policy_sha256']and sha(row['observation_file'])==row['observation_sha256']
 a=load(row['policy_file']);q=row['query'];tick=row['actor_observation_tick']
 assert row['base_infer_calls']==row['actual_internal_noise_calls']==1 and not row['sampled_base_executed']
 assert row['actor_images']=={str(i):array_hash(obs['cam'+str(i)])for i in range(3)}and row['command_state_sha256']==array_hash(obs['command_state'])
 w=row['noise_witness'];assert w['base_sha256']==base_hash(a['base'])and w['tick']==tick and w['environment_unchanged']and w['before']!=w['after']and w['rng_before']!=w['rng_after']and w['shape']==[1,32,32]
 joint=a['joint'][0];state=obs['command_state'];pj,ps=(np.zeros_like(joint),np.zeros_like(state))if prior is None else prior
 expected=pack_observable(joint[None],state[None],pj[None],ps[None],np.asarray([q>0]),projection,robots)
 assert set(expected)==FORWARD_KEYS and (prior is None)==(q==0)
 assert all(np.array_equal(a['legal_'+k],expected[k])for k in expected)and np.array_equal(a['legal_base'],a['base'][None])
 captured={}
 def hook(name):
  def store(module,args,value):captured[name]=value.detach().numpy().copy()
  return store
 hooks=[head.action_head[1].register_forward_hook(hook('hidden')),head.action_head[2].register_forward_hook(hook('logits'))]
 try:
  with torch.inference_mode():out=head({k:torch.from_numpy(v.copy())for k,v in expected.items()},torch.from_numpy(a['base'][None].copy()),enabled=True)
 finally:
  for h in hooks:h.remove()
 assert np.array_equal(out['delta'][0].numpy(),a['raw_delta'])and np.array_equal(captured['hidden'][0],a['hidden'])and np.array_equal(captured['logits'][0],a['logits'])
 h=torch.from_numpy(a['hidden'].copy());l=torch.from_numpy(a['logits'].copy());z=torch.zeros(64)
 assert h.shape==(32,64)and l.shape==(32,14)
 # Keep batch rank identical to the original head, including exact CPU linear execution.
 assert torch.equal(torch.nn.functional.linear(h[None],head.action_head[2].weight,head.action_head[2].bias)[0],l)
 reference=torch.from_numpy(a['raw_delta'].copy())[:,ARM]
 shift=torch.nn.functional.linear((h[None]*z+z),head.action_head[2].weight[ARM])[0]
 zero=(reference+.1*(torch.tanh(l[:,ARM]+shift)-torch.tanh(l[:,ARM]+torch.zeros_like(shift)))).clamp(-.1,.1)
 assert torch.equal(zero,reference)and not np.any(a['raw_delta'][:,[6,13]])
 return a,(joint.copy(),state.copy()),21+len(expected)
def trace_check(chunk,before,after,commands):
 assert sha(chunk['file'])==chunk['sha256'];a=load(chunk['file']);n=chunk['committed_ticks']
 assert np.array_equal(a['source_commands'],commands)and len(commands)==chunk['source_count']
 assert len(a['dt_s'])==len(a['clock_commands'])==n and np.all(abs(a['dt_s']-.004)<1e-8)
 if n:
  for k,d in(('arm_position_target',12),('arm_velocity_target',12),('gripper_logical_command',2)):
   assert a[k].shape==(n,d)and np.isfinite(a[k]).all()
  reconstructed=np.column_stack((a['arm_position_target'][:,:6],a['gripper_logical_command'][:,0],a['arm_position_target'][:,6:],a['gripper_logical_command'][:,1])).astype(np.float32)
  assert np.array_equal(reconstructed,a['clock_commands'])
 count=before['native_action_count'];end=0;valid=[];partial=0
 for i,r in enumerate(chunk['receipts']):
  assert r['offset']==i and r['start_tick']==end and r['end_tick']>=end and r['count_before']==count and r['count_after']in(count,count+1)
  if r['status']=='returned':assert r['count_after']==count+1;valid.append(i)
  elif r['count_after']==count+1:partial+=1
  end=r['end_tick'];count=r['count_after']
 assert end==n and after['native_action_count']==count and after['global_tick']==before['global_tick']+n
 if chunk['reason']=='source_chunk_returned':assert len(chunk['receipts'])==len(commands)and all(r['status']=='returned'for r in chunk['receipts'])
 frames=0
 if 'dense_file'in chunk:
  assert sha(chunk['dense_file'])==chunk['dense_sha256'];d=load(chunk['dense_file']);phase=None
  if chunk['completion']:assert len(chunk['completion'])==1;phase=chunk['completion'][0]['requested95_tick']
  wanted=sorted({i for i in COMMON_TICKS if i<=n}|({phase}if phase is not None else set())|{n})
  assert d['ticks'].tolist()==wanted and len(chunk['readouts'])==len(wanted)==chunk['dense_frame_count']
  assert d['head_rgb'].shape==(len(wanted),240,320,3)and d['head_rgb'].dtype==np.uint8
  assert np.array_equal(d['command_state'],a['clock_commands'][np.asarray(wanted)-1])and np.array_equal(d['elapsed_s'],np.cumsum(a['dt_s'])[np.asarray(wanted)-1])
  for i,r in enumerate(chunk['readouts']):assert r['tick']==wanted[i]and r['elapsed_s']==d['elapsed_s'][i]and r['head_RGB_sha256']==array_hash(d['head_rgb'][i])
  final=load(after['file']);assert np.array_equal(d['head_rgb'][-1],final['cam0'])and np.array_equal(d['command_state'][-1],final['command_state']);frames=len(wanted)
 return a,np.asarray(valid,np.int64),partial,frames,18+4*len(chunk['receipts'])+3*frames

def audit(spec,root,head=None,robots=None,projection=None):
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');assert not torch.cuda.is_initialized();verify(spec)
 from cx_tta_feedback_confirm5.query import build_head
 from cx_tta.multi_event import EventDetector,EventConfig
 from cx_tta.empirical_onset_gate import candidate_packet
 from native_expert_clock_audit_v1.report import phase
 before_rng=torch.random.get_rng_state().clone()
 if head is None:
  with torch.random.fork_rng(devices=[]):head,robots=build_head(spec['training'])
 if projection is None:projection=load(spec['projection'])
 weights={k:v.detach().clone()for k,v in head.state_dict().items()};gpu=read(root/'GPU_native_complete.json')
 assert gpu['complete']and gpu['protocol_identity']==spec['identity']and len(gpu['rows'])==len(spec['scenes'])==96
 counts=dict(requested=96,available=0,queries=0,valid_commands=0,partial_commands_excluded=0,unissued_offsets_excluded=0,dense_frames=0,checks=0)
 records=[];unavailable=[]
 for index,(item,row)in enumerate(zip(spec['scenes'],gpu['rows'],strict=True)):
  assert row['index']==index and row['original_index']==item['index']and row['case_id']==item['case_id']and row['split']==item['split']
  folder=root/('case%03d'%index);assert read(folder/'capture.json')==row
  if not row['available']:
   assert row['reason']in('original_initial_identity_mismatch_retained','native_unstable_initialization_retained');unavailable.append(row);continue
  counts['available']+=1;source=load(item['file'])['absolute_source_commands'];prior=None;detector=EventDetector(EventConfig(**spec['event_config']))
  assert row['matched_initial_object_robot_command_exact']and not row['initialization_eval_mode']
  proof=read(folder/'initial_identity.json');assert proof['offline_diagnostic_only']and all(proof['equal'].values())and sha(proof['actual_file'])==proof['actual_sha256']and sha(proof['source_file'])==proof['source_sha256']
  actual=load(proof['actual_file']);expected=load(proof['source_file']);assert all(np.array_equal(v,expected['first_saved_'+k])for k,v in actual.items())
  assert len(row['observations'])==len(row['chunks'])+1 and len(row['queries'])==len(row['events'])==len(row['chunks'])
  for q,obs in enumerate(row['observations']):
   assert obs['query']==q and sha(obs['file'])==obs['sha256'];a=load(obs['file']);assert set(a)=={'cam0','cam1','cam2','command_state'}
   assert all(a['cam'+str(i)].shape==(240,320,3)and a['cam'+str(i)].dtype==np.uint8 for i in range(3))and a['command_state'].shape==(14,)and np.isfinite(a['command_state']).all()
  for q,(chunk,query,event)in enumerate(zip(row['chunks'],row['queries'],row['events'],strict=True)):
   obs=row['observations'][q];following=row['observations'][q+1]
   assert chunk['query']==query['query']==event['query']==q and chunk['source_start']==q*32 and chunk['global_start_tick']==obs['global_tick']==query['actor_observation_tick']
   assert query['environment_physics_RNG_unchanged']and query['observation_file']==obs['file']and query['observation_sha256']==obs['sha256']
   observed=load(obs['file']);policy,prior,nchecks=policy_check(query,observed,prior,head,robots,projection);counts['checks']+=nchecks
   reconstructed,evidence=detector.step(q,candidate_packet({k.removeprefix('legal_'):v for k,v in policy.items()if k.startswith('legal_')},policy['raw_delta']))
   assert reconstructed==event['event']and evidence==event['evidence']
   a,valid,partial,frames,nchecks=trace_check(chunk,obs,following,source[q*32:q*32+32]);counts['checks']+=nchecks
   if reconstructed is not None and len(reconstructed['kinds'])==1:
    measured=phase(reconstructed['kinds'][0],observed['command_state'],a['source_commands'],a,len(valid),len(valid)==32 and chunk['complete_source_chunk'])
    if chunk['completion']:
     assert measured['available'];expected_phase=measured['record'];actual_phase=chunk['completion'][0]
     for k in('kind','planned_goal','planned_amplitude','planned_base_sha256','onset_tick','requested95_tick','actual_ticks','origin_completed_tick','status'):assert expected_phase[k]==actual_phase[k]
    else:assert not measured['available']or chunk['committed_ticks']==0
   teacher=a['source_commands'][valid][:,ARM].copy();base=policy['base'][valid][:,ARM].copy();target=np.clip(teacher-base,-.1,.1).astype(np.float32)
   values=dict(valid_offsets=valid,teacher_ARM=teacher,once_base_ARM=base,direction_target_ARM=target);file=folder/('target%03d.npz'%q)
   if file.exists():assert all(np.array_equal(v,load(file)[k])for k,v in values.items())
   else:
    with file.open('xb')as f:np.savez_compressed(f,**values)
   records.append(dict(case_id=item['case_id'],original_expert_index=item['index'],split=item['split'],query=q,
    actor_observation_tick=query['actor_observation_tick'],observation_file=obs['file'],observation_sha256=obs['sha256'],
    policy_file=query['policy_file'],policy_sha256=query['policy_sha256'],file=str(file),sha256=sha(file),
    valid_command_count=len(valid),terminal_partial_excluded=partial,unissued_excluded=32-len(valid)-partial,feedback_packet_bound=False))
   counts['queries']+=1;counts['valid_commands']+=len(valid);counts['partial_commands_excluded']+=partial;counts['unissued_offsets_excluded']+=32-len(valid)-partial;counts['dense_frames']+=frames
  assert row['actual_policy_draws']==len(row['queries'])and row['actual_native_action_count']==row['observations'][-1]['native_action_count']and row['committed_ticks']==row['observations'][-1]['global_tick']
 assert counts['queries']==gpu['actual_policy_draws']and counts['available']==gpu['available']
 assert torch.equal(before_rng,torch.random.get_rng_state())and all(torch.equal(weights[k],v)for k,v in head.state_dict().items())and not torch.cuda.is_initialized()
 report=dict(passed=True,counts=counts,records=records,unavailable_cases=unavailable,all96_original_identities_accounted=True,
   original80FIT20VAL_unchanged=True,validity_rule='actually issued AND fully returned only',four_complete_pilot_cases_not_repeated=True,
   actual_query_once_base_hidden64_and_teacher_phase_checked=True,full_original_expert_cache_ready=False,
   semantic_tracking_and_feedback_packets_still_missing=True,training_steps=0,online_updates=0,TTA_utility_proven=False)
 f=root/'CPU_qualification.json'
 if f.exists():assert read(f)==report
 else:write(f,report)
 print('ORIGINAL96_NATIVE_CPU_AUDIT',counts,flush=True);return report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();audit(read(O/'protocol.json'),a.dataset)
