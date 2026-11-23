"""Task-independent, privileged teacher projection; no learned policy or task router."""
import hashlib
import json
from contextlib import contextmanager
import numpy as np

from .native_teacher import SceneStepGuard, export_commands
from .recovery_contract import rng_capture, rng_restore, rng_fingerprint

CAPS = np.array([.1]*6+[.2]+[.1]*6+[.2], np.float32)


class ProjectionStop(Exception):
    pass


class PlanningFailed(Exception):
    pass


def diagnostic_value(value):
    """Small JSON-safe diagnostic values; never inputs to control or inference."""
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray): return diagnostic_value(value.tolist())
    if isinstance(value, np.generic): return diagnostic_value(value.item())
    if isinstance(value, (list, tuple)): return [diagnostic_value(v) for v in value]
    if isinstance(value, dict): return {str(k): diagnostic_value(v) for k,v in value.items()}
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, float): return value if np.isfinite(value) else str(value)
    return str(value)


@contextmanager
def planner_status(task, side):
    """Observe native MotionGen result before RoboTwin collapses it to Fail.

    No extra planner calls or RNG draws. Remote planner workers are explicitly
    marked unavailable instead of pretending that their subtype was observed.
    """
    planner = getattr(task.robot, side+'_planner', None)
    motion = getattr(planner, 'motion_gen', None)
    detail = dict(available=motion is not None, calls=[])
    if motion is None:
        detail['reason'] = 'no in-process MotionGen interface'
        yield detail
        return
    original = motion.plan_single
    local = vars(motion).get('plan_single')
    had_local = 'plan_single' in vars(motion)
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        fields = ('status','success','valid_query','attempts','used_graph',
                  'position_error','rotation_error')
        detail['calls'].append({key:diagnostic_value(getattr(result,key))
                               for key in fields if hasattr(result,key)})
        return result
    motion.plan_single = observed
    try:
        yield detail
    finally:
        if had_local: motion.plan_single = local
        else: del motion.plan_single


class IsolatedQuery:
    """Preserve the model's internal sampler; keep planner draws off its stream."""
    def __init__(self, infer, state):
        self.infer, self.state = infer, state

    def __call__(self, observation):
        ambient = rng_capture()
        before = rng_fingerprint(self.state)
        try:
            rng_restore(self.state)
            actions = self.infer(observation)
            self.state = rng_capture()
            after = rng_fingerprint(self.state)
        finally:
            rng_restore(ambient)
        if rng_fingerprint(rng_capture()) != rng_fingerprint(ambient):
            raise RuntimeError('Policy query contaminated planner RNG')
        return actions, dict(before=before, after=after)


class ProjectionExecutor:
    """One model query per 32 callback attempts, including terminal no-ops."""
    def __init__(self, task, observation, query, *, first_base, first_rng,
                 on_query, on_action, horizon=None, caps=None):
        self.task, self.obs, self.query = task, observation, query
        self.first_base, self.first_rng = first_base, first_rng
        self.on_query, self.on_action = on_query, on_action
        self.horizon, self.caps = horizon, caps
        self.guard = SceneStepGuard(task.scene)
        task.scene = self.guard
        self.prefix, self.limit = int(task.take_action_cnt), int(task.step_lim)
        self.attempts = self.executed = self.queries = self.teacher_actions = 0
        self.base = None
        self.cursor = 32

    @property
    def terminal(self):
        return self.task.eval_success or self.task.take_action_cnt >= self.limit

    def ensure_query(self):
        if self.task.step_lim != self.limit: raise RuntimeError('Budget changed')
        if self.terminal: raise ProjectionStop('native_terminal')
        if self.cursor == 32:
            if self.queries == 0:
                base, rng = self.first_base.copy(), self.first_rng
            else:
                base, rng = self.query(self.obs)
            base = np.asarray(base, np.float32)
            if base.shape != (32,14) or not np.isfinite(base).all():
                raise ValueError('Bad base block')
            self.base, self.cursor = base, 0
            self.on_query(self.queries, self.obs, base, rng)
            self.queries += 1

    def check_teacher(self):
        if self.terminal: raise ProjectionStop('native_terminal')
        if self.horizon is not None and self.executed >= self.horizon:
            raise ProjectionStop('teacher_horizon')

    def step(self, desired=None, metadata=None):
        if desired is not None: self.check_teacher()
        self.ensure_query()
        base = self.base[self.cursor]
        if desired is None:
            action = base.copy()
        else:
            desired = np.asarray(desired, np.float32)
            if desired.shape != (14,) or not np.isfinite(desired).all():
                raise ValueError('Bad teacher command')
            action = desired.copy() if self.caps is None else base+np.clip(desired-base, -self.caps, self.caps)
        self._apply(action, desired, metadata or {}, padding=False)
        if desired is not None: self.teacher_actions += 1
        if self.terminal:
            # Official callbacks still observe after early success/budget exhaustion.
            while self.cursor < 32:
                self._apply(self.base[self.cursor], None, {}, padding=True)
            raise ProjectionStop('native_terminal')
        return action

    def _apply(self, action, desired, metadata, padding):
        count, ticks = self.task.take_action_cnt, self.guard.steps
        self.guard.in_native = True
        try:
            self.task.take_action(action.copy())
        finally:
            self.guard.in_native = False
        self.obs = self.task.get_obs()
        advanced = int(self.task.take_action_cnt)-count
        if advanced != (0 if padding else 1): raise RuntimeError('Native count mismatch')
        self.on_action(dict(query=self.queries-1, slot=self.cursor, action=action.tolist(),
            base=self.base[self.cursor].tolist(), desired=None if desired is None else desired.tolist(),
            advanced=advanced, count=int(self.task.take_action_cnt), padding=padding,
            physics_steps=self.guard.steps-ticks, success=bool(self.task.eval_success), **metadata), self.obs)
        self.cursor += 1
        self.attempts += 1
        self.executed += advanced

    def continue_base(self):
        try:
            while not self.terminal: self.step()
        except ProjectionStop as exc:
            if str(exc) != 'native_terminal': raise


def replay_teacher_prefix(executor, rows, verify_before, verify_after, on_new_plan):
    """Controlled horizon comparison, not open-loop recovery after divergence.

    Only replay a completed long branch's common prefix from the identical fork.
    Reject any predecessor-state mismatch before executing the next action, and
    any reached-state mismatch immediately after it. Never reuse a path beyond
    that common prefix, or turn a mismatch into an experimental failure label.
    """
    if len(rows)!=32: raise ValueError('Expected exactly one 32-action prefix')
    plans=set()
    for i,row in enumerate(rows):
        if row['padding'] or row['advanced']!=1 or row['desired'] is None:
            raise ValueError('Prefix must contain 32 real teacher actions')
        executor.check_teacher()
        executor.ensure_query()
        verify_before(i,row)
        if row['plan'] not in plans:
            on_new_plan(row['plan'])
            plans.add(row['plan'])
        applied=executor.step(np.asarray(row['desired'],np.float32),
            {k:row[k] for k in ('plan','dense_index','phase')})
        if not np.array_equal(applied,np.asarray(row['action'],np.float32)):
            raise RuntimeError('Common teacher prefix action mismatch')
        verify_after(i,row)
    return len(plans)


class ReplanningTeacher:
    """Reuse official semantic goals, replan from actual state at each query.

    A semantic goal is fixed during its phase, as in the official script. No
    stored trajectory is resumed after a query boundary. A phase may advance
    only once its endpoint command was actually issued without clipping the
    active dimensions. This tests following THIS teacher, not existence of all
    possible bounded recovery strategies.
    """
    def __init__(self, task, executor, stride, on_plan, on_attempt=None):
        self.task, self.executor, self.stride, self.on_plan = task, executor, stride, on_plan
        self.phase = self.plans = 0
        self.on_attempt = on_attempt
        self.attempts = 0
        task.move = self.move
        task.take_dense_action = self.unsupported
        task.together_move_to_pose = self.unsupported

    def unsupported(self, *args, **kwargs):
        raise RuntimeError('Unvalidated direct/double-arm teacher execution')

    def move(self, actions_by_arm1, actions_by_arm2=None, save_freq=-1):
        if actions_by_arm2 is not None: self.unsupported()
        if not self.task.plan_success: raise PlanningFailed('official_plan_failed')
        side, actions = actions_by_arm1
        side = str(side)
        if side not in ('left','right'): raise ValueError('Invalid teacher arm')
        for action in actions:
            if str(action.arm_tag) != side: raise ValueError('Arm mismatch')
            self._phase(side, action)
            self.phase += 1
        return True

    def _phase(self, side, action):
        while True:
            self.executor.check_teacher()
            self.executor.ensure_query()
            current = self.task.robot.get_left_arm_jointState()+self.task.robot.get_right_arm_jointState()
            controls = dict(left_arm=None, left_gripper=None, right_arm=None, right_gripper=None)
            offset = 0 if side == 'left' else 7
            if action.action == 'move':
                goal = dict(pose=diagnostic_value(action.target_pose),
                    constraint=diagnostic_value(action.args.get('constraint_pose')))
                paths = getattr(self.task, side+'_joint_path', [])
                previous_paths = len(paths)
                with planner_status(self.task, side) as detail:
                    plan = getattr(self.task, side+'_move_to_pose')(
                        pose=action.target_pose, constraint_pose=action.args.get('constraint_pose'))
                raw = paths[-1] if len(paths)>previous_paths else plan
                status = None if raw is None else raw.get('status')
                if self.on_attempt is not None:
                    self.on_attempt(self.attempts, dict(phase=self.phase, side=side, kind='move',
                        goal=goal, query=self.executor.queries-1, native_count=int(self.task.take_action_cnt),
                        current_command=current, returned_plan=plan is not None,
                        plan_success=bool(self.task.plan_success), status=diagnostic_value(status),
                        native_result_keys=[] if raw is None else sorted(raw), motion_gen=detail))
                self.attempts += 1
                if plan is None or not self.task.plan_success:
                    raise PlanningFailed('motion_plan_failed')
                controls[side+'_arm'] = plan
                active = slice(offset,offset+6)
            elif action.action == 'gripper':
                controls[side+'_gripper'] = self.task.set_gripper(
                    **{side+'_pos':action.target_gripper_pos}, set_tag=side)
                active = slice(offset+6,offset+7)
                goal = dict(gripper=float(action.target_gripper_pos))
            else: raise ValueError('Unknown semantic action')
            dense, indices = export_commands(controls,current,self.stride)
            plan_id = self.plans
            self.plans += 1
            self.on_plan(plan_id, dense, indices, dict(phase=self.phase, side=side,
                kind=action.action, goal=goal, query=self.executor.queries-1,
                native_count=int(self.task.take_action_cnt), current_command=current,
                measured_arm_qpos=[np.asarray(self.task.robot.get_left_arm_real_jointState())[:6].tolist(),
                                   np.asarray(self.task.robot.get_right_arm_real_jointState())[:6].tolist()]))
            completed = False
            for index in indices:
                applied = self.executor.step(dense[index], dict(plan=plan_id, dense_index=int(index),phase=self.phase))
                if index == indices[-1] and np.max(np.abs(applied[active]-dense[index,active])) <= 2e-6:
                    completed = True
                    break
                if self.executor.cursor == 32:
                    break  # next plan starts from actual state, not old remaining path
            if completed: return
