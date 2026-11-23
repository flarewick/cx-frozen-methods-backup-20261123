"""Privileged native-planner constraint inspection and explicit alignment goals.

No task identity or object-state rules. The first verified transition supports
rotation-only hold constraints; other constraints are inspected but not repaired.
"""
import numpy as np
from .teacher_projection import diagnostic_value

ROTATION_METRIC_LIMIT = .05  # native MotionGen metric, NOT radians
POSITION_LIMIT_M = .005     # native partial-position validity threshold


def hold_compatible(rotation_metric, position_error, weights, project_to_goal=True):
    weights = np.asarray(weights, dtype=np.float32).reshape(6)
    position_error = np.asarray(position_error, dtype=np.float32).reshape(3)
    if not np.isfinite(np.r_[rotation_metric, position_error, weights]).all():
        raise ValueError('Non-finite native constraint inputs')
    rotation_ok = not (project_to_goal and (weights[:3]>0).any() and np.float32(rotation_metric)>np.float32(ROTATION_METRIC_LIMIT))
    position_ok = not (np.abs(position_error[weights[3:]>0])>np.float32(POSITION_LIMIT_M)).any()
    return bool(rotation_ok and position_ok)


def quaternion_degrees(first, second):
    a,b = np.asarray(first,float).reshape(4),np.asarray(second,float).reshape(4)
    if not np.isfinite(np.r_[a,b]).all() or min(np.linalg.norm(a),np.linalg.norm(b))<1e-12:
        raise ValueError('Invalid quaternion')
    return float(np.degrees(2*np.arccos(np.clip(abs(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),0,1))))


def alignment_goal(measured_pose, requested_goal, weights):
    """Hold the current native EE-reference position and align orientation.

    This is an explicit extra phase, not an unconstrained substitute for the
    original approach. It must be measured and counted before approach resumes.
    """
    current,goal = np.asarray(measured_pose,float),np.asarray(requested_goal,float)
    w=np.asarray(weights,float)
    if current.shape!=(7,) or goal.shape!=(7,) or w.shape!=(6,) or not np.isfinite(np.r_[current,goal,w]).all():
        raise ValueError('Invalid pose or constraint')
    if not (w[:3]>0).any() or np.any(w[3:]!=0):
        raise ValueError('Only rotation-only hold alignment is validated')
    quaternion_degrees(current[3:],goal[3:])
    return np.r_[current[:3],goal[3:]/np.linalg.norm(goal[3:])].tolist()


class _CapturedQuery(Exception):
    pass


def capture_native_query(robot, side, pose, constraint):
    """Run native frame conversion/rounding, stop before invoking the solver.

    Do not duplicate RoboTwin's frame transforms. No task plan_success reset,
    trajectory search, simulation step or global/installed source edit occurs.
    """
    if side not in ('left','right') or getattr(robot,'communication_flag',False):
        raise ValueError('In-process single-arm native planner required')
    motion=getattr(robot,side+'_planner').motion_gen
    original=motion.plan_single
    local=vars(motion).get('plan_single');had_local='plan_single' in vars(motion)
    captured=[]
    def capture(start,goal,config,*args,**kwargs):
        captured.append((motion,start,goal,config))
        raise _CapturedQuery()
    motion.plan_single=capture
    try:
        try:getattr(robot,side+'_plan_path')(pose,constraint_pose=constraint)
        except _CapturedQuery:pass
    finally:
        if had_local:motion.plan_single=local
        else:del motion.plan_single
    if len(captured)!=1 or motion.plan_single!=original:
        raise RuntimeError('Native query interception/restoration failed')
    return captured[0]


def inspect_native_constraint(robot, side, pose, constraint):
    motion,start,goal,config=capture_native_query(robot,side,pose,constraint)
    measured=motion.compute_kinematics(start).ee_pose.clone()
    if str(measured.quaternion.dtype)!='torch.float32':
        raise ValueError('Constraint parity currently validated for native float32 only')
    metric=config.pose_cost_metric
    project=True if metric is None else metric.project_to_goal_frame
    if project is None:project=motion.get_all_pose_rollout_instances()[0].goal_cost.project_distance
    projected=goal.compute_local_pose(measured)
    identity=type(goal).from_list([0,0,0,1,0,0,0],tensor_args=motion.tensor_args)
    rotation=float(projected.angular_distance(identity).detach().cpu().max().item())
    position=projected.position if project else goal.position-measured.position
    position=np.asarray(diagnostic_value(position),float).reshape(-1,3)[0]
    weights=np.zeros(6) if metric is None else np.asarray(diagnostic_value(metric.hold_vec_weight),float).reshape(6)
    active=bool(metric is not None and metric.hold_partial_pose and metric.offset_tstep_fraction<0)
    compatible=not active or hold_compatible(rotation,position,weights,bool(project))
    mq=np.asarray(diagnostic_value(measured.quaternion),float).reshape(-1,4)[0]
    gq=np.asarray(diagnostic_value(goal.quaternion),float).reshape(-1,4)[0]
    return dict(compatible=compatible,active=active,project_to_goal_frame=bool(project),comparison_dtype='float32',
        rotation_metric=rotation,rotation_metric_limit=ROTATION_METRIC_LIMIT,
        position_error_m=position.tolist(),position_limit_m=POSITION_LIMIT_M,weights=weights.tolist(),
        rotation_geodesic_degrees=quaternion_degrees(mq,gq),
        planner_start_qpos=diagnostic_value(start.position),
        measured_planner_pose=dict(position=diagnostic_value(measured.position),quaternion=diagnostic_value(measured.quaternion)),
        goal_planner_pose=dict(position=diagnostic_value(goal.position),quaternion=diagnostic_value(goal.quaternion)),
        measured_native_ee_pose=diagnostic_value(getattr(robot,'get_'+side+'_ee_pose')()),
        requested_world_pose=diagnostic_value(pose),requested_constraint=diagnostic_value(constraint))
