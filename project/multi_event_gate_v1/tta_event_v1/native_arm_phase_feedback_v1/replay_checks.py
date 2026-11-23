"""Actual qualified TRAIN20 replay plus unequal arms/censor/future-step rejection."""
from pathlib import Path
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from .completion import ArmPathCompletion,ARM

def reject(fn):
 try:fn()
 except ValueError:return
 raise AssertionError('Invalid phase accepted')

def main():
 initial=np.zeros(14,np.float32);commands=np.zeros((32,14),np.float32);commands[:,0]=np.linspace(0,1,32);commands[:,7]=np.linspace(0,1,32)
 phase=ArmPathCompletion(initial,commands);reject(lambda:phase.after_arm_actuation(np.zeros(12),committed_tick=1,dt_s=.004));reject(lambda:phase.command_begin(1));phase.command_begin(0);reject(lambda:phase.command_begin(0))
 # A fast left arm never certifies the lagging right arm.
 for i in range(32):
  if i:phase.command_begin(i)
  target=commands[i,ARM].copy();target[6]=0
  phase.after_arm_actuation(target,committed_tick=i+1,dt_s=.004)
 assert phase.side95[0]is not None and phase.side95[1]is None and phase.pending['event']['requested95_tick']is None
 phase.finish_origin('origin',arrived_tick=32,commands_returned=32,native_censored=False);assert phase.closed[0]['status']=='censored'
 phase=ArmPathCompletion(initial,commands)
 for i in range(31):
  phase.command_begin(i);phase.after_arm_actuation(commands[i,ARM],committed_tick=i+1,dt_s=.004)
 reject(lambda:phase.observe_rgb('origin',observation_tick=31,decision_tick=31,frame_id='x'))
 phase.finish_origin('origin',arrived_tick=31,commands_returned=31,native_censored=True);assert phase.closed[0]['status']=='censored'
 source=T/'outputs/native_onpolicy_timing_audit_v1/qualified_TRAIN20_roller_timing.json';audit=read(source);assert audit['qualified_TRAIN_only']and audit['actual_unique_roller_cases']==20
 bycase={r['case_id']:r for r in audit['records']};rows=[];inputs={str(source):sha(source)}
 for name in audit['input_sha256']:
  p=mapped(name)
  if p.name!='result.json'or p.parent.name!='M_event_full':continue
  result=read(p);q=result['queries'][0];tracefile=mapped(q['execution']['actual_trace_file']);meta=mapped(p.parent/'execution000.json');obs=mapped(q['observation']['file'])
  for f in(p,tracefile,meta,obs):inputs[str(f)]=sha(f)
  with np.load(tracefile,allow_pickle=False)as z:trace={k:z[k].copy()for k in('arm_position_target','source_commands','dt_s')}
  with np.load(obs,allow_pickle=False)as z:state=z['command_state'].copy()
  phase=ArmPathCompletion(state,trace['source_commands']);receipt=read(meta);last=0
  for command in receipt['receipts']:
   assert command['start_tick']==last
   phase.command_begin(command['offset'])
   for tick in range(command['start_tick'],command['end_tick']):phase.after_arm_actuation(trace['arm_position_target'][tick],committed_tick=tick+1,dt_s=trace['dt_s'][tick])
   last=command['end_tick']
  returned=sum(c['status']=='returned'for c in receipt['receipts']);phase.finish_origin('origin',arrived_tick=last,commands_returned=returned,native_censored=not receipt['complete_source_chunk']);phase.observe_rgb('origin',observation_tick=last,decision_tick=last,frame_id='actual_TRAIN_trace_timing_only')
  value=phase.closed[0];tick=value['requested95_tick'];lag=None if tick is None else float(trace['dt_s'][tick:].sum())
  rows.append(dict(case_id=p.parent.parent.name,actual_origin_ticks=last,side95_ticks=phase.side95,requested95_tick=tick,actual_lag_before_q1_s=lag,actual32_and_all_sides95=value['status']=='feedback_ready',potential_q1_time_only=bool(value['status']=='feedback_ready'and lag>=.25),actual_new_visual_feedback_available=False))
 assert len(rows)==20 and {r['case_id']for r in rows}==set(bycase)
 write(T/'outputs/native_arm_phase_feedback_v1/CPU_actual_TRAIN20_phase_checks.json',dict(passed=True,actual_qualified_TRAIN_replays=20,all_moving_arms_latest95_required=True,missing_native32_and_unequal_sides_rejected=True,no_query_or_scene_advance=True,original_phase_lag_and_cooldown_NOT_changed=True,potential_q1_timing=sum(r['potential_q1_time_only']for r in rows),timing_only_no_live_RGB_teacher_or_TTA_coverage_claim=True,records=rows,input_sha256=inputs,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()}))
 print('CPU_ACTUAL_TRAIN20_ALL_ARM_PHASE_REPLAY_PASS',sum(r['potential_q1_time_only']for r in rows))
if __name__=='__main__':main()
