"""Independently reconstruct source-command/tick/trace alignment, retaining terminal cases."""
from pathlib import Path
import argparse,numpy as np
from native_expert_frames_v1.paths import *
def audit(spec,root):
 verify(spec);value=read(root/'GPU_frames_complete.json');assert value['complete']and value['protocol_identity']==spec['identity']and len(value['rows'])==len(spec['scenes'])==4
 checks=0;summary=[];dense_frames=0;phase_events=0
 from native_stage_v1.clock import COMMON_TICKS
 phase_report=read(T/'outputs/native_expert_clock_audit_v1/report.json');phases={(r['case_id'],r['query']):r for r in phase_report['rows']}
 for item,row in zip(spec['scenes'],value['rows']):
  assert item['case_id']==row['case_id'];checks+=1
  if not row['available']:
   assert row['reason']=='native_unstable_initialization_retained';summary.append(dict(case_id=item['case_id'],available=False));continue
  with np.load(item['file'],allow_pickle=False)as z:source=z['absolute_source_commands'].copy()
  assert row['matched_initial_RGB_command_object_robot_exact']and not row['initialization_eval_mode']and row['actual_boundary_RGB_state_physics_RNG_exact'];checks+=1
  original=item['reference_capture'];assert len(row['chunks'])==len(original['chunks'])and row['native_success_diagnostic_only']==original['native_success_diagnostic_only']
  observations=row['observations'];assert len(observations)==len(row['chunks'])+1;count=0;tick=0
  for q,obs in enumerate(observations):
   assert obs['query']==q and sha(obs['file'])==obs['sha256']
   with np.load(obs['file'],allow_pickle=False)as a:
    assert set(a.files)=={'cam0','cam1','cam2','command_state'}
    assert all(a['cam'+str(i)].shape==(240,320,3)and a['cam'+str(i)].dtype==np.uint8 for i in range(3))
    assert a['command_state'].shape==(14,)and np.isfinite(a['command_state']).all()
   reference=original['observations'][q]
   with np.load(reference['file'],allow_pickle=False)as z,np.load(obs['file'],allow_pickle=False)as a:
    assert set(z.files)==set(a.files)and all(np.array_equal(a[k],z[k])for k in a.files)
   for key in ('global_tick','native_action_count','physical_sha256','rng_sha256'):assert obs[key]==reference[key];checks+=1
   assert obs['reference_query_exact'];checks+=7
  for q,r in enumerate(row['chunks']):
   assert r['query']==q and r['source_start']==q*32 and r['global_start_tick']==tick and sha(r['file'])==r['sha256']
   assert observations[q]['global_tick']==tick and observations[q]['native_action_count']==count
   with np.load(r['file'],allow_pickle=False)as z:
    a={k:z[k].copy()for k in z.files}
   assert np.array_equal(a['source_commands'],source[q*32:q*32+32])and len(a['source_commands'])==r['source_count'];checks+=3
   n=r['committed_ticks'];assert len(a['dt_s'])==len(a['clock_commands'])==n
   old=original['chunks'][q]
   with np.load(old['file'],allow_pickle=False)as z:assert set(z.files)==set(a)and all(np.array_equal(z[k],a[k])for k in a)
   assert r['receipts']==old['receipts']and r['reason']==old['reason'];checks+=2
   enabled=q in item['dense_queries'];assert ('dense_file'in r)==enabled;checks+=1
   if enabled:
    assert sha(r['dense_file'])==r['dense_sha256']
    with np.load(r['dense_file'],allow_pickle=False)as z:frames={k:z[k].copy()for k in z.files}
    phase=None
    if str(q)in item['event_kinds']:
     assert len(r['completion'])==1;phase_events+=1
     actual=r['completion'][0];expected=phases[item['case_id'],q]['actual_executed_plan_phase']['record']
     for key in ('kind','planned_goal','planned_amplitude','planned_base_sha256','onset_tick','requested95_tick','actual_ticks','origin_completed_tick','status'):
      assert actual[key]==expected[key],(item['case_id'],q,key);checks+=1
     phase=actual['requested95_tick']
    else:assert r['completion']==[]
    wanted=sorted(set(k for k in COMMON_TICKS if k<=n)|({phase}if phase is not None else set())|{n})
    assert frames['ticks'].tolist()==wanted and r['dense_frame_count']==len(wanted)and len(r['readouts'])==len(wanted);checks+=3
    assert frames['head_rgb'].shape==(len(wanted),240,320,3)and frames['head_rgb'].dtype==np.uint8
    assert np.array_equal(frames['command_state'],a['clock_commands'][np.asarray(wanted)-1])
    assert np.array_equal(frames['elapsed_s'],np.cumsum(a['dt_s'])[np.asarray(wanted)-1]);checks+=3
    from cx_tta.eval_diagnostics import array_hash
    for i,(k,meta)in enumerate(zip(wanted,r['readouts'])):
     assert meta['tick']==k and meta['elapsed_s']==frames['elapsed_s'][i]
     assert meta['head_RGB_sha256']==array_hash(frames['head_rgb'][i]);checks+=2
    with np.load(observations[q+1]['file'],allow_pickle=False)as z:
     assert np.array_equal(frames['head_rgb'][-1],z['cam0'])and np.array_equal(frames['command_state'][-1],z['command_state']);checks+=2
    dense_frames+=len(wanted)
   if n:
    for key,dim in(('arm_position_target',12),('arm_velocity_target',12),('gripper_logical_command',2)):
     assert a[key].shape==(n,dim)and np.isfinite(a[key]).all();checks+=1
    assert np.all(abs(a['dt_s']-.004)<1e-8)
    reconstructed=np.column_stack((a['arm_position_target'][:,:6],a['gripper_logical_command'][:,0],a['arm_position_target'][:,6:],a['gripper_logical_command'][:,1])).astype(np.float32)
    assert np.array_equal(reconstructed,a['clock_commands']);checks+=2
   prev=0
   for i,v in enumerate(r['receipts']):
    assert v['offset']==i and v['start_tick']==prev and v['end_tick']>=prev and v['count_before']==count
    assert v['count_after']in(count,count+1);count=v['count_after'];prev=v['end_tick'];checks+=3
   assert prev==n and (r['reason']!='source_chunk_returned'or len(r['receipts'])==r['source_count'])
   tick+=n;assert observations[q+1]['global_tick']==tick and observations[q+1]['native_action_count']==count;checks+=3
  assert count==row['actual_native_action_count']and tick==row['committed_ticks']
  summary.append(dict(case_id=item['case_id'],available=True,source_commands=len(source),native_actions=count,committed_ticks=tick,
   executed_time_s=float(tick*np.float32(.004)),queries=len(row['chunks']),source_exhausted=row['source_exhausted'],native_success_diagnostic_only=row['native_success_diagnostic_only'],
   terminal_reason=row['chunks'][-1]['reason']if row['chunks']else'no_native_actions'))
 assert phase_events==16
 return dict(passed=True,checks=checks,requested=4,rows=summary,CPU_only=True,actual_velocity_and_command_clock_verified=True,
  all_original_drive_clock_query_RGB_physics_RNG_exact=True,dense_arrived_head_frames=dense_frames,actual_expert_phase_events=phase_events,
  no_new_VLA_draws=True,causal_SAM2_local_feedback_not_yet_computed=True,
  native_training_cache_ready=False,feedback_packets_ready=False,once_base_hidden64_ready=False,online_updates=0,TTA_utility_proven=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();s=read(O/'protocol.json')
 assert read(a.dataset/'result.json')['complete'];v=audit(s,a.dataset);write(a.dataset/'interface_qualification.json',v)
 print('EXPERT_NATIVE_CPU_AUDIT',json.dumps(v),flush=True)
if __name__=='__main__':main()
