"""Privileged measurement-gated native teacher, with explicit alignment phases."""
import numpy as np
from .constraint_teacher import inspect_native_constraint, alignment_goal, quaternion_degrees
from .teacher_projection import ReplanningTeacher, PlanningFailed, planner_status, diagnostic_value
from .native_teacher import export_commands

POSITION_TOLERANCE_M=.005
ORIENTATION_TOLERANCE_DEG=3.

def measured_goal(robot,side,pose):
    measured=np.asarray(getattr(robot,'get_'+side+'_ee_pose')(),float)
    goal=np.asarray(pose,float)
    distance=float(np.linalg.norm(measured[:3]-goal[:3]))
    angle=quaternion_degrees(measured[3:],goal[3:])
    return dict(pose=measured.tolist(),distance_m=distance,angle_degrees=angle,
        reached=bool(distance<=POSITION_TOLERANCE_M and angle<=ORIENTATION_TOLERANCE_DEG))

class CompatibleTeacher(ReplanningTeacher):
    def __init__(self,*args,on_measurement,**kwargs):
        super().__init__(*args,**kwargs)
        self.on_measurement=on_measurement

    def measure(self,side,action,stage):
        robot=self.task.robot
        result=dict(phase=self.phase,side=side,kind=action.action,stage=stage,
                    native_count=int(self.task.take_action_cnt))
        if action.action=='move':
            result['goal']=diagnostic_value(action.target_pose)
            result['arrival']=measured_goal(robot,side,action.target_pose)
            constraint=action.args.get('constraint_pose')
            result['constraint']=None if constraint is None else inspect_native_constraint(robot,side,action.target_pose,constraint)
        else:
            # get_*_gripper_val is a command, NOT measured finger aperture.
            entity=getattr(robot,side+'_entity');joints=entity.get_active_joints();qpos=entity.get_qpos()
            fingers=getattr(robot,side+'_gripper')
            result.update(target=float(action.target_gripper_pos),command=float(getattr(robot,'get_'+side+'_gripper_val')()),
                finger_qpos=[float(qpos[joints.index(j[0])]) for j in fingers if j is not None],
                completion_rule='endpoint command; measured fingers logged, no unsupported contact/closure certificate')
        self.on_measurement(result)
        return result

    def _phase(self,side,action):
        alignment=None
        while True:
            self.executor.check_teacher();self.executor.ensure_query()
            measurement=self.measure(side,action,'before_plan')
            mask=action.args.get('constraint_pose') if action.action=='move' else None
            incompatible=mask is not None and not measurement['constraint']['compatible']
            if action.action=='move' and measurement['arrival']['reached'] and not incompatible:return
            current=self.task.robot.get_left_arm_jointState()+self.task.robot.get_right_arm_jointState()
            controls=dict(left_arm=None,left_gripper=None,right_arm=None,right_gripper=None)
            offset=0 if side=='left' else 7
            mode='alignment' if incompatible else 'original'
            if action.action=='move':
                if incompatible:
                    if alignment is None:alignment=alignment_goal(measurement['arrival']['pose'],action.target_pose,mask)
                    target,requested_mask=alignment,None
                else:
                    alignment=None;target,requested_mask=action.target_pose,mask
                # Original native planner; do not poison/reset task wrapper flag.
                with planner_status(self.task,side) as detail:
                    plan=getattr(self.task.robot,side+'_plan_path')(target,constraint_pose=requested_mask)
                ok=plan['status']=='Success'
                self.on_attempt(self.attempts,dict(phase=self.phase,side=side,kind='move',mode=mode,
                    goal=dict(pose=diagnostic_value(target),constraint=diagnostic_value(requested_mask)),
                    original_goal=dict(pose=diagnostic_value(action.target_pose),constraint=diagnostic_value(mask)),
                    query=self.executor.queries-1,native_count=int(self.task.take_action_cnt),current_command=current,
                    returned_plan=ok,plan_success=bool(self.task.plan_success),status=plan['status'],
                    native_result_keys=sorted(plan),motion_gen=detail))
                self.attempts+=1
                if not ok:raise PlanningFailed('compatible_teacher_'+mode+'_planning_failed')
                controls[side+'_arm']=plan;active=slice(offset,offset+6)
                goal=dict(pose=diagnostic_value(target),constraint=diagnostic_value(requested_mask))
            elif action.action=='gripper':
                controls[side+'_gripper']=self.task.set_gripper(**{side+'_pos':action.target_gripper_pos},set_tag=side)
                active=slice(offset+6,offset+7);goal=dict(gripper=float(action.target_gripper_pos))
            else:raise ValueError('Unknown semantic action')
            dense,indices=export_commands(controls,current,self.stride)
            pid=self.plans;self.plans+=1
            self.on_plan(pid,dense,indices,dict(phase=self.phase,side=side,kind=action.action,mode=mode,goal=goal,
                query=self.executor.queries-1,native_count=int(self.task.take_action_cnt),current_command=current))
            for index in indices:
                applied=self.executor.step(dense[index],dict(plan=pid,dense_index=int(index),phase=self.phase,mode=mode))
                after=self.measure(side,action,'after_action')
                if action.action=='move':
                    compatible=after['constraint'] is None or after['constraint']['compatible']
                    if after['arrival']['reached'] and compatible:return
                    if mode=='alignment' and compatible:break
                    if mode=='original' and not compatible:break
                elif index==indices[-1] and np.max(np.abs(applied[active]-dense[index,active]))<=2e-6:return
                if self.executor.cursor==32:break
