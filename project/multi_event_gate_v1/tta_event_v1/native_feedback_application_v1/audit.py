"""Independently reconstruct actual fast states/actions from saved arrived packets."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,numpy as np,torch
from native_policy_execution_v1.common import read,sha,write
from native_response_binding_v1.bind import inputs,ARM,tensor
from native_response_action_v1.compose import EventComposer
from cx_tta.multi_event import apply_arm_candidate
from native_shared_response_v1.session import state_sha
from stage_feedback_v3.completion import CompletionQueue as OriginalCompletion

def arrays(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def digest(fast):return state_sha(dict(gain=fast[0],offset=fast[1]))
def audit(spec,root,*,response_model,predictor,output=None):
 previous_threads=torch.get_num_threads();torch.set_num_threads(2)
 try:
  model=copy.deepcopy(response_model).eval();model.response.requires_grad_(False);records=[];joint_checks=[];single_checks=[]
  for folder in sorted((root/'cases').glob('*/*')):
   if not(folder/'result.json').exists():continue
   result=read(folder/'result.json');condition=result['condition'];updates={u['query']:u for u in result['feedback']['updates']};assert len(updates)<=3;fast=tuple(v.detach()for v in model.zero_fast());anchors=[];sources={read(p)['source_query']:read(p)for p in folder.glob('source*/source.json')};policy={q['query']:arrays(q['policy']['policy_file'])for q in result['queries']};issuing={q['query']:dict(query=q['query'],tick=q['tick'],hidden=policy[q['query']]['hidden'])for q in result['queries']};active=[];checked=0
   for row in result['queries']:
    q=row['query'];value=policy[q];reference=tensor(value['raw_delta'][:,ARM][None]);h=tensor(value['hidden'][None]);logits=tensor(value['logits'][:,ARM][None]);u=updates.get(q)
    if u is not None:
     assert u['fast_before']==digest(fast);source=sources[u['source_query']];ids=next(b['eligible_rows']for b in source['boundaries']if b['query']==q);local=arrays(source['local_file']);rows={i:{k:local[k][i]for k in local}for i in range(len(local['target']))};proof='CPU_actual_packet_reconstruction';packet=SimpleNamespace(rows=tuple(ids),source_query=u['source_query'],query=q,observation_tick=row['tick'],query_proof_sha256=proof);current=dict(value,query=q,tick=row['tick'],query_proof_sha256=proof);v=inputs(packet,rows=rows,issuing_queries=issuing,current=current,predictor=predictor)
     proposed,_=model.two_steps(**{k:v[k]for k in('hidden','context','command','kind','kernel_prediction','target')},fast=fast,online=True)
     with torch.no_grad():
      args={k:v[k]for k in('hidden','context','command','kind','kernel_prediction')};before=float(((model(**args,fast=fast)-v['target'])/.003125).square().mean());after=float(((model(**args,fast=proposed)-v['target'])/.003125).square().mean())
     trusted,_=model.trust(h,logits,reference,proposed,anchors);accepted=bool(trusted and digest(proposed)!=digest(fast)and after<before)
     assert accepted==u['accepted']and abs(before-u['response_loss_before'])<1e-6 and abs(after-u['response_loss_after'])<1e-6
     if accepted:fast=tuple(t.detach()for t in proposed);anchors=[tuple(t.detach().clone()for t in(h,logits,reference)),*anchors][:2]
     assert digest(fast)==u['fast_after']
    with torch.no_grad():
     trusted,_=model.trust(h,logits,reference,fast,anchors);direction=model.direction(h,logits,reference,fast).numpy()[0]if trusted and condition=='TTA'else value['raw_delta'][:,ARM]
    event=row['event'];is_active=event['decision']['active'];selected=event['decision']['selected_event'];expected=np.zeros_like(value['base'])
    if is_active:
     if selected is not None:expected[:,ARM]=direction
     else:
      witness=event['feedback_fast_difference'];assert condition=='TTA'and witness['applied'];accepted_query=witness['accepted_query'];assert accepted_query in updates and updates[accepted_query]['accepted']and updates[accepted_query]['fast_after']==witness['fast_after']==digest(fast)and accepted_query<=q<=accepted_query+2
      expected[:,ARM]=direction-value['raw_delta'][:,ARM];assert np.max(np.abs(expected))<=.020001
     if active:assert q-active[-1]>spec['event_config']['cooldown_queries']
     active.append(q);assert len(active)==event['decision']['interventions']<=3
    actual=arrays(row['actions_file']);expected_actions,expected_delta=apply_arm_candidate(value['base'],expected,is_active)
    assert np.array_equal(expected_delta,actual['applied_delta'])and np.array_equal(expected_actions,actual['actions']),('Actual action is not reconstructed fast-only composition',folder,q)
    checked+=1
   semantic=read(folder/'semantic_initial_receipt.json');count=semantic.get('actual_object_count')
   for source in sources.values():
    if count==1:
     masks=arrays(source['mask_file']);assert not masks['masks'][:,1].any()and not masks['presence'][:,1].any();assert all(r['object_index']==0 for r in source['rows'])
     if source['local_file']is not None:
      local=arrays(source['local_file']);assert not local['relative_available'].any()and np.isnan(local['relative_target']).all()and not local['context'][:,-9:].any()
     single_checks.append(dict(folder=str(folder),source=source['source_query'],actual_one_ID_only=True))
    origin=read(folder/('execution%03d.json'%source['source_query']));completion=origin['completion']
    if completion and completion[0].get('all_sides_required'):
     joint=completion[0];trace=arrays(origin['actual_trace_file']);obs=arrays(result['queries'][source['source_query']]['observation']['file']);individual=[]
     for kind in joint['kinds']:
      queue=OriginalCompletion();queue.register('event','origin',kind,obs['command_state'][[6,13]],trace['source_commands'],decision_tick=0,frame_id='CPU_reconstruction')
      for tick,(grip,dt)in enumerate(zip(trace['gripper_logical_command'],trace['dt_s'],strict=True),1):queue.after_actuation('origin',grip,committed_tick=tick,dt_s=float(dt))
      phase=queue.pending['event']['requested95_tick'];individual.append(phase)
     expected=max(individual)if all(p is not None for p in individual)and origin['complete_source_chunk']else None
     assert joint['requested95_tick']==expected;assert [r['requested95_tick']for r in joint['components']]==individual
     joint_checks.append(dict(folder=str(folder),source=source['source_query'],actual_component95_ticks=individual,latest_actual_tick=expected))
   records.append(dict(folder=str(folder),condition=condition,actual_query_actions_reconstructed=checked,accepted_updates=sum(u['accepted']for u in updates.values()),actual_active_queries=active,fast_updates_reconstructed_exact=True))
  proof=dict(passed=True,actual_native_branches=len(records),records=records,joint_actual_phase_reconstruction=joint_checks,single_real_ID_checks=single_checks,original_two_SGD_and_actual_fast_digests_reconstructed=True,all_actual_actions_equal_frozen_full_or_trusted_fast_difference=True,total_three_and_cooldown_checked=True,no_new_GPU_work=True,CUDA_initialized=torch.cuda.is_initialized(),no_task_efficacy_claim=True)
  write(output or root/'CPU_feedback_application_audit.json',proof);return proof
 finally:torch.set_num_threads(previous_threads)
