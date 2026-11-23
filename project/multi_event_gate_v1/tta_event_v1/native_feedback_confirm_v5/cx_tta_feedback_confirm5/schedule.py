"""Explicit bounded diagnostic commands on a common native Z schedule.

Only commanded drive targets are changed, just before the actual scene step.
Never read measured positions/velocities, contacts, objects, outcomes or rewards.
This is NOT native TOPP execution of the original production candidate.
"""
from contextvars import ContextVar
import numpy as np
from .paths import read
ACTIVE=ContextVar('cx_native_stage_v1',default=None)
DOSES=dict(Z_command_replay=0.,P_positive_quarter=.25,P_negative_quarter=-.25,P_positive_half=.5,P_negative_half=-.5)
ARM=[0,1,2,3,4,5,7,8,9,10,11,12]


def desired(reference,raw,dose):
    if dose not in DOSES.values():raise ValueError('Undeclared bounded dose')
    if raw.shape!=(32,14) or raw.dtype!=np.float32 or not np.isfinite(raw).all() or np.any(raw[:,[6,13]]) or np.max(abs(raw))>.100001:
        raise ValueError('Invalid original frozen residual')
    dt=np.asarray(reference['dt_s']);t=np.cumsum(dt);ramp=.5
    phase=np.minimum(t/ramp,1.)
    f=.5*(1-np.cos(np.pi*phase));df=np.where(t<ramp,.5*np.pi/ramp*np.sin(np.pi*phase),0.)
    direction=raw[0,ARM].astype(float)
    dp=dose*f[:,None]*direction;dv=dose*df[:,None]*direction
    if np.max(abs(dp))>.050001 or np.max(abs(dv))>.157081:raise ValueError('Probe command bounds exceeded')
    result={k:v.copy() for k,v in reference.items()}
    # Explicit float32 matches SAPIEN one-DOF drive storage; zero adds nothing.
    if dose:
        for k,d in [('arm_position_target',dp),('arm_velocity_target',dv)]:
            result[k]=(reference[k]+d).astype(np.float32).astype(np.float64)
    return result


class Plan:
    def __init__(self,reference,raw,dose):self.targets=desired(reference,raw,dose);self.index=0
    @classmethod
    def from_branch(cls,branch,folder,raw):
        if branch not in DOSES:return None
        with np.load(folder/'Z_reference_actuation.npz',allow_pickle=False) as f:ref={k:f[k].copy() for k in f.files}
        return cls(ref,raw,DOSES[branch])
    def apply(self,robot,dt):
        i=self.index;t=self.targets
        if i>=500 or dt!=t['dt_s'][i]:raise ValueError('Command override shifted native tick')
        arms=tuple(robot.left_arm_joints)+tuple(robot.right_arm_joints)
        fingers=[j for side in (robot.left_gripper,robot.right_gripper) for j,_,_ in side]
        if len(arms)!=12 or len(fingers)!=t['gripper_position_target'].shape[1]:raise ValueError('Joint mapping changed')
        for joints,position,velocity in [(arms,'arm_position_target','arm_velocity_target'),(fingers,'gripper_position_target','gripper_velocity_target')]:
            for j,p,v in zip(joints,t[position][i],t[velocity][i]):
                j.set_drive_target(np.array([p],np.float32));j.set_drive_velocity_target(np.array([v],np.float32))
        robot.left_gripper_val,robot.right_gripper_val=map(np.float64,t['gripper_logical_command'][i])
        self.index+=1


def trace_step(task):
    from .prefix_time import ACTIVE as PREFIX
    from actuation_telemetry import trace_step as original
    from cx_tta_clock.clock import ACTIVE as CLOCK
    plan=ACTIVE.get()
    if plan is not None:
        clock=CLOCK.get()
        if clock is None or plan.index!=len(clock.dt):raise ValueError('Probe requires original shared physical clock')
        plan.apply(task.robot,float(task.scene.get_timestep()))
    counter=PREFIX.get()
    if counter is not None and CLOCK.get() is None:
        dt=float(task.scene.get_timestep());value=original(task);counter.append(1,dt);return value
    from stage_feedback_bridge_v1.bridge import ACTIVE as FEEDBACK
    hook=FEEDBACK.get()
    return original(task) if hook is None else hook.call(original,task)


def instrument(source):
    from actuation_telemetry import instrument as original
    text=original(source);needle='from actuation_telemetry import trace_step as _cx_diag_step\n'
    if text.count(needle)!=1:raise ValueError('Frozen trace instrumentation changed')
    return text.replace(needle,'from cx_tta_feedback_confirm5.schedule import trace_step as _cx_diag_step\n')


def verify_traces(traces,raw,packets):
    z=traces['Z_reference']
    for branch,dose in DOSES.items():
        expected=desired(z,raw,dose);got=traces[branch]
        if set(got)!=set(expected) or any(not np.array_equal(got[k],expected[k]) for k in got):
            raise ValueError('Shared-clock full-command schedule mismatch: '+branch)
    for k in packets['Z_reference']:
        if not np.array_equal(packets['Z_command_replay'][k],packets['Z_reference'][k]):
            raise ValueError('Zero command replay changed native payload: '+k)
