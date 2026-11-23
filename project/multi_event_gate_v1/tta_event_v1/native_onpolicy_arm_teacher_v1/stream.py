"""Provisional same-state labels; only a qualified atomic parent can admit them."""
from pathlib import Path
import argparse,importlib,weakref,gc
import numpy as np
from native_policy_execution_v1.common import write
from native_onpolicy_recovery_v3.common import read,sha,mapped
from native_onpolicy_recovery_v3.state import arrays,require_same
from native_reference_recovery_v1.selector import select
from native_onpolicy_full_v1.index import index
ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12])

def packet(spec,folder,item):
 if spec.get('initial_ARM_path_feedback'):assert spec['actual_seed_start']in(7124000,7125000)
 assert spec['scope']=='frozen_full_onpolicy_TRAIN_capture'and spec['actual_seed_start']in(7120000,7121000,7124000,7125000)
 assert spec['actual_seed_start']<=int(item['case_id'].split(':')[1])<spec['actual_seed_start']+1000
 result=read(folder/'result.json');assert result['complete']and result['condition']=='M_event_full'
 actions=index(folder,spec['event_config'])['potential_actions']
 if not actions:return None
 chosen=actions[0];q=chosen['action_query'];row=result['queries'][q];ob=row['observation'];policy=row['policy'];env=read(folder/f'query{q:03d}_environment_audit.json')
 assert sha(mapped(ob['file']))==ob['sha256']and sha(mapped(policy['policy_file']))==policy['policy_sha256']
 sources=[read(p)['source_query']for p in folder.glob('source*/source.json')];prefix=[]
 for i,r in enumerate(result['queries'][:q]):
  execution=r['execution'];assert execution['actual_complete_native32']and execution['commands_returned']==32
  assert sha(mapped(r['actions_file']))==r['actions_sha256']and sha(mapped(execution['actual_trace_file']))==execution['actual_trace_sha256']
  commands=arrays(r['actions_file'])['actions'];actual=arrays(execution['actual_trace_file']);assert np.array_equal(commands,actual['source_commands'])
  kind=None
  if i in sources:
   admission=r.get('feedback_source_admission',{});kind=admission.get('kind')
   if kind is None:
    kinds=r['event']['decision']['selected_event']['kinds'];kind=kinds[0]if len(kinds)==1 else tuple(kinds)
   elif isinstance(kind,list):kind=tuple(kind)
  prefix.append(dict(actions=commands,metadata=read(folder/f'execution{i:03d}.json'),actual_trace=actual,kind=kind))
 source_kind=result['queries'][chosen['packet']['source_query']].get('feedback_source_admission',{}).get('kind')
 return dict(source_kind=source_kind,query=q,prefix=prefix,choice=chosen,expected_observation=arrays(ob['file']),once_policy=arrays(policy['policy_file']),expected_native_action_count=q*32,expected_physical=env['physical_sha256'],expected_rng=env['rng_sha256'])

def recover(spec,root,case,c,prepared,bridge,number,*,lane):
 from cx_tta_feedback_confirm5.lanes import Lane
 from cx_tta_feedback_confirm5.throughput import replay_prefix
 from cx_tta.collect_smoke import make_config
 from cx_tta.evaluate_b2 import seed_all
 from cx_tta import recovery_contract as recovery
 from native_initial_frame_v3.observe import physical
 from native_expert_smoke_v2.observe import observe
 from native_arm_phase_feedback_v1.capture import capture
 import torch
 full=root/'cases'/case['case_id']/'M_event_full';p=packet(spec,full,case)
 if p is not None:
  from stage_feedback_v3.completion import KINDS
  source=read(full/'result.json')['queries'][p['choice']['packet']['source_query']]
  kinds=(source['event']['decision']['selected_event']or{}).get('kinds',[])
  joint=len(kinds)==2 and all(k in KINDS for k in kinds)and len({KINDS[k][0]for k in kinds})==2
  if joint or p.get('source_kind')=='initial_own_ARM_path':
   from .live import recover as live_recover
   return live_recover(spec,root,case,c,prepared,bridge,number,lane=lane,p=p)
 folder=root/'recoveries'/case['case_id'];folder.mkdir(parents=True)
 row=dict(case_id=case['case_id'],task=case['task'],label_available=False,provisional_until_entire_parent_batch_qualifies=True,optimizer_steps=0,no_actor_or_online_teacher=True)
 if p is None:
  row['reason']='no_legal_feedback_after_full_action';write(folder/'result.json',row);return row
 ref=read(root/'expert_references'/(case['case_id']+'.json'));assert ref['case_id']==case['case_id']and ref['expert_plan_success']and ref['expert_task_success']and ref['sampling_stride_physics_steps']==15 and sha(mapped(ref['file']))==ref['sha256']
 reference=arrays(ref['file'])['absolute_source_commands'];row.update(query=p['query'],application_mode=p['choice']['application_mode'],reference_file=ref['file'],reference_sha256=ref['sha256'],packet=p['choice']['packet'])
 assert c.rng.current() is lane,'Recovery must reuse its actual completed-full worker lane'
 cfg=make_config(prepared['runtime'],argparse.Namespace(task=case['task'],task_config='demo_randomized'),prepared['session']);cfg.update(eval_mode=True,save_data=False,eval_video_log=False,eval_video_save_dir=None,render_freq=0)
 with c.rng.bind(lane):
  def create():
   from shared6_task_bookkeeping import install_task
   task=getattr(importlib.import_module('envs.'+case['task']),case['task'])();install_task(task);return c.prepare_task(task,lane)
  task=c.call(lane,create);raw=lambda:c.call(lane,observe,task);seed=c.routed(seed_all);primary_error=False
  def check(value,label):
   diagnostic=c.call(lane,lambda:dict(physical_sha256=physical(task),rng_sha256=recovery.rng_fingerprint(recovery.rng_capture())))
   proof=require_same(value,diagnostic,p);file=folder/(label+'_state.npz')
   with file.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):v for i,v in enumerate(value['rgb'])},command_state=value['command_state'],native_action_count=np.asarray(value['native_action_count']))
   write(folder/(label+'_same_state.json'),dict(**proof,**diagnostic,state_file=str(file),state_sha256=sha(file)))
  try:
   current=replay_prefix(task,case,[],cfg,raw,seed)
   for i,part in enumerate(p['prefix']):
    bridge.begin_case(case['case_id']+'/recovery_prefix'+str(i),number)
    try:actual,meta,_=capture(task,part['actions'],reader=raw,initial=current,kind=part['kind'],collect_dense=bool(part['metadata']['dense_frame_count']),on_arrival=None,check_cancel=c.check_cancel)
    finally:readout=bridge.finish_case()
    original=part['metadata'];assert actual.keys()==part['actual_trace'].keys()and all(np.array_equal(actual[k],part['actual_trace'][k],equal_nan=True)for k in actual)
    assert all(meta[k]==original[k]for k in('receipts','reason','committed_ticks','readouts'))and readout['native_calls']==meta['committed_ticks'];current=raw()
    write(folder/f'prefix{i:03d}_exact.json',dict(passed=True,actual_drive_physics_readout_cadence_exact=True))
   check(current,'before_selection');suffix,selection=select(reference,current['command_state']);write(folder/'reference_selection.json',selection);current=raw();check(current,'before_execution')
   bridge.begin_case(case['case_id']+'/recovery_execution',number);ticks=0;returned=0;first=None
   for i,start in enumerate(range(0,len(suffix),32)):
    source=suffix[start:start+32].copy();trace,meta,_=capture(task,source,reader=raw,initial=current,kind=None,collect_dense=False,on_arrival=None,check_cancel=c.check_cancel)
    file=folder/f'teacher_execution{i:03d}.npz'
    with file.open('xb')as f:np.savez_compressed(f,**trace)
    write(folder/f'teacher_execution{i:03d}.json',dict(**meta,actual_trace_file=str(file),actual_trace_sha256=sha(file)))
    valid=np.asarray([r['offset']for r in meta['receipts']if r['status']=='returned'],np.int64);ticks+=meta['committed_ticks'];returned+=len(valid)
    if i==0:first=(source,valid)
    if meta['reason']!='source_chunk_returned':break
    current=raw()
   readout=bridge.finish_case();assert readout['native_calls']==ticks;write(folder/'teacher_readout_bridge.json',readout)
   native_success=c.call(lane,lambda:bool(task.eval_success));task_success=c.call(lane,lambda:bool(task.check_success()));source,valid=first
   row.update(native_teacher_replay_success=native_success,original_task_success=task_success,returned_commands=returned,first_action_returned=len(valid),reference_start_index=selection['reference_start_index'])
   if native_success and task_success and len(valid):
    target=folder/'teacher_target.npz';base=p['once_policy']['base']
    with target.open('xb')as f:np.savez_compressed(f,valid_offsets=valid,teacher_ARM=source[valid][:,ARM],once_base_ARM=base[valid][:,ARM],direction_target_ARM=np.clip(source[valid][:,ARM]-base[valid][:,ARM],-.1,.1).astype(np.float32))
    row.update(label_available=True,target_file=str(target),target_sha256=sha(target),reason='same_state_native_recovery_success',teacher_vs_base_first_GRIP_difference_diagnostic_only=float(np.max(np.abs(source[valid][:,[6,13]]-base[valid][:,[6,13]]))),online_GRIP_remains_frozen=True)
   else:row['reason']='reference_guided_native_recovery_failed_retained'
  except BaseException:
   import traceback
   primary_error=True;write(folder/'primary_failure.json',dict(traceback=traceback.format_exc(),no_dataset_qualification=True));raise
  finally:
   task.close_env(clear_cache=False);task_ref=weakref.ref(task);scene_ref=weakref.ref(task.scene);task=None;raw=None
   def release():
    with c.scene_gate.write():gc.collect();torch.cuda.empty_cache()
    if not primary_error:assert task_ref()is None and scene_ref()is None
   c.call(lane,release)
 write(folder/'result.json',row);print('PROVISIONAL_FULL_NATIVE_RECOVERY',case['case_id'],row['label_available'],row['reason'],flush=True);return row

def audit(spec,root):
 records=[]
 for case in read(root/'GPU_episodes_complete.json')['cases']:
  folder=root/'recoveries'/case['case_id'];row=read(folder/'result.json');assert row['provisional_until_entire_parent_batch_qualifies']and row['optimizer_steps']==0
  if row.get('teacher_kind')=='live_current_geometry_open_replan':
   from .live import audit as live_audit
   p=packet(spec,root/'cases'/case['case_id']/'M_event_full',case);assert p is not None;live_audit(spec,root,case,row,p);records.append(row);continue
  if 'query'in row:
   p=packet(spec,root/'cases'/case['case_id']/'M_event_full',case)
   for label in('before_selection','before_execution'):
    proof=read(folder/(label+'_same_state.json'));a=arrays(proof['state_file']);assert sha(mapped(proof['state_file']))==proof['state_sha256'];require_same(dict(rgb=[a[f'cam{i}']for i in range(3)],command_state=a['command_state'],native_action_count=int(a['native_action_count'])),proof,p)
   assert sha(mapped(row['reference_file']))==row['reference_sha256'];suffix,selection=select(arrays(row['reference_file'])['absolute_source_commands'],p['expected_observation']['command_state']);assert selection['reference_start_index']==row['reference_start_index']
   meta=read(folder/'teacher_execution000.json');actual=arrays(meta['actual_trace_file']);assert sha(mapped(meta['actual_trace_file']))==meta['actual_trace_sha256']and np.array_equal(actual['source_commands'],suffix[:32]);valid=np.asarray([r['offset']for r in meta['receipts']if r['status']=='returned'],np.int64)
   assert row['label_available']==bool(row['native_teacher_replay_success']and row['original_task_success']and len(valid))
   if row['label_available']:
    target=arrays(row['target_file']);assert sha(mapped(row['target_file']))==row['target_sha256']and np.array_equal(target['valid_offsets'],valid)and np.array_equal(target['teacher_ARM'],suffix[valid][:,ARM])and np.array_equal(target['once_base_ARM'],p['once_policy']['base'][valid][:,ARM])and np.array_equal(target['direction_target_ARM'],np.clip(target['teacher_ARM']-target['once_base_ARM'],-.1,.1).astype(np.float32))
  if not row['label_available']:assert not(folder/'teacher_target.npz').exists()
  records.append(row)
 assert len(records)==spec['cases_per_batch']
 groups={t:dict(cases=sum(r['task']==t for r in records),potential_states=sum(r['task']==t and 'query'in r for r in records),labels=sum(r['task']==t and r['label_available']for r in records))for t in spec['tasks']}
 result=dict(passed=True,records=records,groups=groups,coverage_passed=all(g['labels']>0 for g in groups.values()),optimizer_steps=0,parent_terminal_and_actual_host_exit_still_required=True)
 write(root/'CPU_recovery_stream_audit.json',result);return result
