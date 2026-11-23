"""Shared native-action probes and passive control-loop timing (no task GT)."""
import ast
import inspect
import textwrap

import numpy as np


SHARED_BRANCHES = ('base', 'plus', 'minus', 'basis_plus', 'basis_minus',
                   'random_plus', 'random_minus')


def shared_candidates(base, caps, *, scene_seed, chunk, strength=0.25):
    """One native coordinate + energy-matched random direction, each signed.

    The coordinate rotates deterministically across seed/chunk, identically for
    every task. Native channels are NOT Cartesian directions. No absolute
    command clipping is introduced; the simulator retains its original command
    semantics, as for B2. Bounds below are residual bounds only.
    """
    base, caps = np.asarray(base, np.float32), np.asarray(caps, np.float32)
    if (base.shape != (32, 14) or caps.shape != (14,)
            or not np.isfinite(base).all() or not np.isfinite(caps).all()
            or np.any(caps <= 0) or not 0 < strength <= 1
            or scene_seed < 0 or chunk < 0):
        raise ValueError('Invalid native intervention inputs')
    index = (int(scene_seed) + int(chunk)) % 14
    basis = np.eye(14)[index]
    # Private generator: never perturb the simulator's global random stream.
    rng = np.random.default_rng(np.random.SeedSequence([int(scene_seed), int(chunk), 271828]))
    random = rng.normal(size=14)
    random /= np.linalg.norm(random)
    envelope = np.sin(np.linspace(0, np.pi, 32)) ** 2
    envelope[[0, -1]] = 0
    proposals = {}
    for name, direction in (('basis', basis), ('random', random)):
        delta = (strength * envelope[:, None] * direction[None] * caps).astype(np.float32)
        proposals[name + '_plus'] = base + delta
        proposals[name + '_minus'] = base - delta
    metadata = {'family': 'shared_native_v1', 'coordinate_index': index,
                'strength_fraction_of_caps': strength, 'caps': caps.tolist(),
                'random_direction_normalized': random.tolist(),
                'energy_matching': 'L2 after dividing residual by per-channel caps',
                'absolute_action_clipping': False}
    return proposals, metadata


def clocked_step(task):
    dt = float(task.scene.get_timestep())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid physics timestep')
    task.scene.step()
    task._cx_intervention_ticks = getattr(task, '_cx_intervention_ticks', 0) + 1
    task._cx_intervention_time = getattr(task, '_cx_intervention_time', 0.0) + dt


def install_action_clock(task_cls):
    """Instrument this process's method only; source runtime stays untouched.

    Fail closed if the inspected official method has an unexpected structure.
    Timing measures its direct physics control loop, not wall time.
    """
    original = task_cls.take_action
    if original.__closure__:
        raise ValueError('Cannot instrument a closure')
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)

    class ReplaceStep(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == 'self.scene.step':
                if node.args or node.keywords:
                    raise ValueError('Unexpected scene.step arguments')
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='_cx_clocked_step', ctx=ast.Load()),
                                                 args=[ast.Name(id='self', ctx=ast.Load())], keywords=[]), node)
            return node

    transform = ReplaceStep()
    tree = transform.visit(tree)
    if transform.count != 1 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Expected exactly one direct physics step in take_action')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _cx_clocked_step=clocked_step)
    exec(compile(tree, inspect.getsourcefile(original), 'exec'), namespace)
    task_cls.take_action = namespace[original.__name__]
    return {'direct_step_sites': transform.count, 'scope': 'take_action direct physics loop',
            'original_source': source, 'instrumented_source': ast.unparse(tree)}


def validate_transition(arrays):
    """Reject malformed/zero-time training samples; preserve early-stop masks."""
    actions = arrays['proposed_actions']
    mask, dt = arrays['executed_mask'], arrays['physics_dt_s']
    ticks = arrays['physics_steps']
    if (actions.shape != (32, 14) or not np.isfinite(actions).all()
            or mask.shape != (32,) or mask.dtype != np.bool_
            or dt.shape != (32,) or not np.isfinite(dt).all() or np.any(dt < 0)
            or ticks.shape != (32,) or ticks.dtype.kind not in 'iu' or np.any(ticks < 0)
            or np.any(dt[~mask] != 0) or np.any(ticks[~mask] != 0)
            or np.any((ticks == 0) != (dt == 0))
            or np.any(np.diff(mask.astype(int)) > 0)):
        raise ValueError('Invalid branch action/timing contract')
    for name in ('head', 'right', 'left'):
        im = arrays[name + '_rgb_after']
        if im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3:
            raise ValueError('Invalid endpoint RGB')
    state = arrays['command_state_after']
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Invalid endpoint command state')
    return {'executed': int(mask.sum()), 'physics_steps': int(ticks.sum()),
            'duration_s': float(dt.sum()), 'positive_time': bool(dt.sum() > 0),
            'full_horizon': bool(mask.all())}
