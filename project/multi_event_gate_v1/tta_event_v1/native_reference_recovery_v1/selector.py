"""Fixed offline reference suffix; labels require subsequent native recovery."""
import numpy as np


def select(commands, command_state):
    commands = np.asarray(commands)
    state = np.asarray(command_state)
    if (commands.ndim != 2 or commands.shape[1] != 14 or not len(commands)
            or state.shape != (14,) or not np.isfinite(commands).all()
            or not np.isfinite(state).all()):
        raise ValueError('Finite actual14 reference and same-state command required')
    # One fixed rule, earliest tie; no success search, threshold or task IDs.
    distances = np.square(commands.astype(np.float64)-state.astype(np.float64)).sum(1)
    index = int(np.argmin(distances))
    return commands[index:].copy(), dict(
        reference_start_index=index, reference_commands=len(commands),
        distance=float(np.sqrt(distances[index])),
        selection='first argmin squared Euclidean actual14 commanded joint/gripper state',
        no_outcome_or_task_selection=True,
        NOT_a_recovery_label_until_native_same_state_execution_succeeds=True)
