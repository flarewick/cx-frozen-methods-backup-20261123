"""Diagnostic bridge: official expert plans -> budgeted native qpos commands.

This privileged teacher is not a deployable adapter. No task-specific recovery
rules are added. Dual-arm direct execution is deliberately unsupported here.
"""
import numpy as np


class TeacherStopped(Exception):
    pass


class SceneStepGuard:
    """Reject expert physics steps that bypass the native budgeted executor."""
    def __init__(self, scene):
        self.scene = scene
        self.in_native = False
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self):
        if not self.in_native:
            raise RuntimeError('Unbudgeted expert scene.step rejected')
        result = self.scene.step()
        self.steps += 1
        return result


def export_commands(control, current, stride):
    """Sample fixed planner indices, retain endpoint and inactive arm targets.

    The stride is measured in dense planner indices, NOT native actions or a
    guaranteed physical duration. Native TOPP determines actual execution time.
    """
    current = np.asarray(current, dtype=np.float64)
    if stride < 1 or current.shape != (14,) or not np.isfinite(current).all():
        raise ValueError('Invalid current command or stride')
    lengths = []
    for side in ('left', 'right'):
        arm, grip = control[side+'_arm'], control[side+'_gripper']
        if arm is not None:
            p = np.asarray(arm['position'])
            if p.ndim != 2 or p.shape[1] != 6 or not len(p) or not np.isfinite(p).all():
                raise ValueError('Invalid arm plan')
            lengths.append(len(p))
        if grip is not None:
            p = np.asarray(grip['result'])
            if p.ndim != 1 or len(p) != grip['num_step'] or not len(p) or not np.isfinite(p).all():
                raise ValueError('Invalid gripper plan')
            if (p < 0).any() or (p > 1).any():
                raise ValueError('Gripper plan outside native normalized range')
            lengths.append(len(p))
    if not lengths:
        raise ValueError('Empty expert plan')
    length = max(lengths)
    dense = np.tile(current, (length, 1))
    for side, offset in (('left', 0), ('right', 7)):
        arm, grip = control[side+'_arm'], control[side+'_gripper']
        if arm is not None:
            dense[:, offset:offset+6] = np.asarray(arm['position'])[np.minimum(np.arange(length), len(arm['position'])-1)]
        if grip is not None:
            dense[:, offset+6] = np.asarray(grip['result'])[np.minimum(np.arange(length), grip['num_step']-1)]
    indices = np.unique(np.r_[np.arange(stride-1, length, stride), length-1])
    return dense.astype(np.float32), indices


class NativeTeacher:
    def __init__(self, task, stride, on_action, on_plan):
        self.task, self.stride = task, stride
        self.on_action, self.on_plan = on_action, on_plan
        self.native_limit = task.step_lim
        self.guard = SceneStepGuard(task.scene)
        task.scene = self.guard
        task.take_dense_action = self.execute
        task.together_move_to_pose = self.unsupported
        self.plans = 0
        self.executed = 0

    def unsupported(self, *args, **kwargs):
        raise RuntimeError('Direct dual-arm expert executor is not validated by this probe')

    def check_terminal(self):
        if self.task.step_lim != self.native_limit:
            raise RuntimeError('Native action limit changed')
        if self.task.eval_success:
            raise TeacherStopped('native_success')
        if self.task.take_action_cnt >= self.native_limit:
            raise TeacherStopped('native_budget_exhausted')

    def execute(self, control_seq, save_freq=-1):
        self.check_terminal()
        current = self.task.robot.get_left_arm_jointState() + self.task.robot.get_right_arm_jointState()
        dense, indices = export_commands(control_seq, current, self.stride)
        plan_id = self.plans
        self.plans += 1
        self.on_plan(plan_id, dense, indices, control_seq)
        for index in indices:
            self.check_terminal()
            count, tick = self.task.take_action_cnt, self.guard.steps
            self.guard.in_native = True
            try:
                self.task.take_action(dense[index].copy())
            finally:
                self.guard.in_native = False
            obs = self.task.get_obs()
            if self.task.take_action_cnt != count+1:
                raise RuntimeError('Native action did not advance once')
            self.executed += 1
            self.on_action(plan_id, int(index), dense[index], obs,
                           self.guard.steps-tick, index == indices[-1])
            self.check_terminal()
        return True
