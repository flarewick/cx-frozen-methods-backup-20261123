"""Passive native physics diagnostics for attribution, never policy inputs.

This captures readable state, not hidden PhysX solver/contact warm-start state.
It does not step, render, query task success, sample RNG, or import SAPIEN.
"""
from collections import Counter

import numpy as np


def _numbers(value, shape=None):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError("Expected finite numeric native state")
    if shape is not None and value.shape != shape:
        raise ValueError(f"Unexpected native shape {value.shape}; expected {shape}")
    return value.tolist()


def _pose(value):
    return _numbers(np.concatenate((np.asarray(value.p), np.asarray(value.q))), (7,))


def _bool_getter(value):
    # SAPIEN 3 exposes is_sleeping as a property, some earlier wrappers as a method.
    return bool(value() if callable(value) else value)


def physical_snapshot(task):
    """Return all readable scene bodies/articulations with fail-closed coverage.

    Keys use scene enumeration and names rather than allocation/global IDs.
    Duplicate names remain distinct and are reported. `usable` means every
    enumerated body/articulation was captured, not complete simulator cloning.
    Failed getters retain an explicit error and cannot silently count as parity.
    """
    errors, actors, articulations = [], [], []
    snapshot = {
        "schema": 1,
        "scope": "offline_only readable native physics; excludes hidden solver/contact state",
        "usable": False,
        "actors": actors,
        "articulations": articulations,
        "control": {},
        "coverage": {},
    }

    def read(label, fn):
        try:
            return fn()
        except Exception as error:
            errors.append({"field": label, "error": f"{type(error).__name__}: {error}"})
            return None

    scene = getattr(task, "scene", None)
    snapshot["timestep"] = read("scene.timestep", lambda: float(scene.get_timestep()))
    if snapshot["timestep"] is not None and (
            not np.isfinite(snapshot["timestep"]) or snapshot["timestep"] <= 0):
        errors.append({"field": "scene.timestep", "error": "Invalid positive finite timestep"})
    native_actors = read("scene.actors", lambda: list(scene.get_all_actors()))
    native_articulations = read("scene.articulations", lambda: list(scene.get_all_articulations()))
    for index, actor in enumerate(native_actors or []):
        prefix = f"actors[{index}]"
        row = {"scene_index": index}
        row["name"] = read(prefix + ".name", lambda: str(actor.get_name()))
        row["pose"] = read(prefix + ".pose", lambda: _pose(actor.get_pose()))
        components = read(prefix + ".components", lambda: list(actor.get_components()))
        bodies = []
        for component in components or []:
            classes = {cls.__name__ for cls in type(component).__mro__}
            if classes.intersection({"PhysxRigidDynamicComponent", "PhysxRigidStaticComponent"}):
                bodies.append(component)
        if len(bodies) != 1:
            errors.append({"field": prefix + ".body", "error": f"Expected one native rigid body; got {len(bodies)}"})
            row["body_type"] = None
        else:
            body = bodies[0]
            classes = {cls.__name__ for cls in type(body).__mro__}
            dynamic = "PhysxRigidDynamicComponent" in classes
            row["body_type"] = "dynamic" if dynamic else "static"
            if dynamic:
                row["linear_velocity"] = read(prefix + ".linear_velocity", lambda: _numbers(body.get_linear_velocity(), (3,)))
                row["angular_velocity"] = read(prefix + ".angular_velocity", lambda: _numbers(body.get_angular_velocity(), (3,)))
                row["sleeping"] = read(prefix + ".sleeping", lambda: _bool_getter(body.is_sleeping))
                row["kinematic"] = read(prefix + ".kinematic", lambda: bool(body.get_kinematic()))
                if row["kinematic"]:
                    row["kinematic_target"] = read(prefix + ".kinematic_target", lambda: _pose(body.get_kinematic_target()))
        actors.append(row)

    for index, articulation in enumerate(native_articulations or []):
        prefix = f"articulations[{index}]"
        row = {"scene_index": index, "active_joints": []}
        row["name"] = read(prefix + ".name", lambda: str(articulation.get_name()))
        row["root_pose"] = read(prefix + ".root_pose", lambda: _pose(articulation.get_root_pose()))
        for field in ("root_linear_velocity", "root_angular_velocity"):
            row[field] = read(prefix + "." + field,
                lambda field=field: _numbers(getattr(articulation, "get_" + field)(), (3,)))
        dof = read(prefix + ".dof", lambda: int(articulation.get_dof()))
        row["dof"] = dof
        for field in ("qpos", "qvel", "qf"):
            row[field] = read(prefix + "." + field,
                lambda field=field: _numbers(getattr(articulation, "get_" + field)(), (dof,)))
        joints = read(prefix + ".active_joints", lambda: list(articulation.get_active_joints()))
        for joint_index, joint in enumerate(joints or []):
            label = f"{prefix}.active_joints[{joint_index}]"
            joint_row = {"index": joint_index}
            joint_row["name"] = read(label + ".name", lambda: str(joint.get_name()))
            joint_dof = read(label + ".dof", lambda: int(joint.get_dof()))
            joint_row["dof"] = joint_dof
            for field in ("drive_target", "drive_velocity_target"):
                joint_row[field] = read(label + "." + field,
                    lambda field=field: _numbers(getattr(joint, "get_" + field)(), (joint_dof,)))
            row["active_joints"].append(joint_row)
        if dof is not None and sum(j["dof"] or 0 for j in row["active_joints"]) != dof:
            errors.append({"field": prefix + ".joint_dof_coverage", "error": "Active joint DOFs do not cover articulation DOFs"})
        articulations.append(row)

    for field in ("take_action_cnt", "step_lim", "eval_success"):
        snapshot["control"][field] = read("control." + field,
            lambda field=field: bool(getattr(task, field)) if field == "eval_success" else int(getattr(task, field)))
    for field in ("stage_success_tag", "plan_success"):
        if hasattr(task, field):
            snapshot["control"][field] = bool(getattr(task, field))
    robot = getattr(task, "robot", None)
    for side in ("left", "right"):
        snapshot["control"][side + "_command"] = read("control." + side + "_command",
            lambda side=side: _numbers(getattr(robot, "get_" + side + "_arm_jointState")()))

    snapshot["coverage"] = {
        "actor_count": None if native_actors is None else len(native_actors),
        "articulation_count": None if native_articulations is None else len(native_articulations),
        "duplicate_actor_names": sorted(name for name, n in Counter(r["name"] for r in actors).items() if n > 1 and name is not None),
        "duplicate_articulation_names": sorted(name for name, n in Counter(r["name"] for r in articulations).items() if n > 1 and name is not None),
        "errors": errors,
        "limitations": [
            "Stable scene enumeration is checked through names/order; allocation IDs are omitted.",
            "This excludes hidden contact/solver caches, planner internal state, and renderer state.",
            "Physics diagnostics never enter adapter/VLA inputs or condition selection.",
        ],
    }
    if not articulations:
        errors.append({"field": "scene.articulations", "error": "No robot articulation captured"})
    snapshot["usable"] = not errors
    return snapshot
