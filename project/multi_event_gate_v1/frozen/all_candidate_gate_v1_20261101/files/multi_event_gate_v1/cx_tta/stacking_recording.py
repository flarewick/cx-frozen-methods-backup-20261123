"""Stable offline actor identities for expert recording, independent of task name.

Task bookkeeping aliases may appear or change during play_once. They are not
new physical objects. Freeze unique actor identities at the first observation;
retain original canonical poses and reject genuine changes to the entity set.
"""
import numpy as np

from .recording import measured_state


def actor_bindings(task):
    for name, value in vars(task).items():
        if name.startswith('_'):
            continue
        values = list(enumerate(value)) if isinstance(value, (list, tuple)) else [(None, value)]
        for index, obj in values:
            if hasattr(obj, 'config') and hasattr(obj, 'actor') and callable(getattr(obj, 'get_pose', None)):
                yield name if index is None else f'{name}_{index}', obj


class ActorRegistry:
    def __init__(self, task):
        self.canonical = {}
        self.identities = {}
        self.aliases = {}
        for name, obj in actor_bindings(task):
            identity = id(obj.actor)
            if identity not in self.identities:
                if name in self.canonical:
                    raise ValueError('Ambiguous canonical actor field: ' + name)
                self.canonical[name] = obj
                self.identities[identity] = name
            else:
                self.aliases.setdefault(name, set()).add(self.identities[identity])
        if not self.canonical:
            raise ValueError('No canonical actors in first observation')

    def poses(self, task):
        current = {}
        for name, obj in actor_bindings(task):
            identity = id(obj.actor)
            if identity not in self.identities:
                raise ValueError('New physical actor appeared during recording: ' + name)
            canonical = self.identities[identity]
            current[name] = identity
            if name != canonical:
                self.aliases.setdefault(name, set()).add(canonical)
        # A changed/deleted canonical binding is not silently treated as an alias.
        for name, obj in self.canonical.items():
            if current.get(name) != id(obj.actor):
                raise ValueError('Canonical physical actor binding changed: ' + name)
        result = {}
        for name, obj in self.canonical.items():
            pose = obj.get_pose()
            result[name] = np.asarray(list(pose.p)+list(pose.q), np.float32)
        return result

    def articulations(self):
        result = {}
        for name, obj in self.canonical.items():
            if not callable(getattr(obj, 'get_qpos', None)):
                continue
            q = np.asarray(obj.get_qpos(), np.float32)
            if q.size == 0:
                continue
            limits = np.asarray(obj.get_qlimits(), np.float32)
            if q.ndim != 1 or limits.shape != (len(q), 2) or not np.isfinite(q).all() or not np.isfinite(limits).all():
                raise ValueError('Invalid canonical articulation state: ' + name)
            result[name] = dict(qpos=q, qlimits=limits)
        return result

    def metadata(self):
        return dict(schema=1, canonical=list(self.canonical),
                    aliases={k: sorted(v) for k, v in self.aliases.items()},
                    identity='In-process actor identity, never equality of pose or task-specific names',
                    role='offline_only_not_policy_inputs')


def install_recording(task, *, effect_schema=False):
    setup, get_obs = task.setup_demo, task.get_obs

    def recorded_setup(*args, **kwargs):
        task._cx_tick = 0
        task._cx_dense = []
        task._cx_actor_registry = None
        return setup(*args, **kwargs)

    def recorded_obs(*args, **kwargs):
        obs = get_obs(*args, **kwargs)
        arms, grippers = measured_state(task.robot)
        obs['cx_measurement'] = dict(physics_tick=np.asarray(task._cx_tick, dtype=np.int64),
            physics_dt_s=np.asarray(task.scene.get_timestep(), dtype=np.float64),
            arm_qpos=arms, gripper_joint_qpos=grippers)
        if task._cx_actor_registry is None:
            task._cx_actor_registry = ActorRegistry(task)
        registry = task._cx_actor_registry
        obs['cx_privileged'] = registry.poses(task)
        if effect_schema:
            obs['cx_effect'] = dict(articulation=registry.articulations())
        return obs

    task.setup_demo = recorded_setup
    task.get_obs = recorded_obs
    task.cx_recording_enabled = True
    task.cx_effect_enabled = effect_schema
