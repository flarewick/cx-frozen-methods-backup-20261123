"""Passive clocks/measurements on a COPIED runtime, never extra rendering."""
import ast
from pathlib import Path
import numpy as np


def instrument_runtime(runtime):
    path = runtime / "envs/_base_task.py"
    text = path.read_text()
    count = text.count("self.scene.step()")
    if count < 1 or "_cx_record_step" in text:
        raise ValueError("Unexpected source or repeated instrumentation")
    text = "from cx_tta.recording import counted_step as _cx_record_step\n" + text.replace(
        "self.scene.step()", "_cx_record_step(self)")
    ast.parse(text)
    path.write_text(text)
    return count


def counted_step(task):
    tick = getattr(task, "_cx_tick", 0)
    command = None
    if getattr(task, "save_data", False) and hasattr(task, "robot"):
        command = np.asarray(task.robot.get_left_arm_jointState() +
                             task.robot.get_right_arm_jointState(), dtype=np.float32)
    task.scene.step()
    task._cx_tick = tick + 1
    if command is not None:
        task._cx_dense.append((tick, float(task.scene.get_timestep()), command))


def measured_state(robot):
    """Return actual arm qpos and raw gripper joint qpos, separate from commands."""
    arms, grippers = [], []
    for arm in ("left", "right"):
        entity = getattr(robot, f"{arm}_entity")
        joints = list(entity.get_active_joints())
        qpos = entity.get_qpos()
        arms.append([float(qpos[joints.index(j)]) for j in getattr(robot, f"{arm}_arm_joints")])
        master = getattr(robot, f"{arm}_gripper")[0][0]
        grippers.append(float(qpos[joints.index(master)]))
    return np.asarray(arms, dtype=np.float32), np.asarray(grippers, dtype=np.float32)


def object_poses(task):
    """Offline privileged actor poses only, never model features."""
    poses = {}
    for name, value in vars(task).items():
        if name.startswith("_"):
            continue
        values = list(enumerate(value)) if isinstance(value, (tuple, list)) else [(None, value)]
        for index, obj in values:
            if hasattr(obj, "config") and hasattr(obj, "actor") and callable(getattr(obj, "get_pose", None)):
                pose = obj.get_pose()
                key = name if index is None else f"{name}_{index}"
                poses[key] = np.asarray(list(pose.p) + list(pose.q), dtype=np.float32)
    return poses


def install_recording(task, *, effect_schema=False):
    setup, get_obs = task.setup_demo, task.get_obs

    def recorded_setup(*args, **kwargs):
        task._cx_tick = 0
        task._cx_dense = []
        return setup(*args, **kwargs)

    def recorded_obs(*args, **kwargs):
        obs = get_obs(*args, **kwargs)
        arms, grippers = measured_state(task.robot)
        obs["cx_measurement"] = {
            "physics_tick": np.asarray(task._cx_tick, dtype=np.int64),
            "physics_dt_s": np.asarray(task.scene.get_timestep(), dtype=np.float64),
            "arm_qpos": arms, "gripper_joint_qpos": grippers,
        }
        obs["cx_privileged"] = object_poses(task)
        if effect_schema:
            from .effect_contract import articulation_state
            obs["cx_effect"] = {"articulation": articulation_state(task)}
        return obs

    task.setup_demo = recorded_setup
    task.get_obs = recorded_obs
    task.cx_recording_enabled = True
    task.cx_effect_enabled = effect_schema
