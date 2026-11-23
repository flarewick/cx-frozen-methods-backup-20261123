"""Narrow handling of RoboTwin's missing grasp target during expert screening.

The simulator and policy rollout remain untouched. Unexpected errors still abort.
"""
import traceback
from pathlib import Path


class ExpertPlanningFailure(Exception):
    def __init__(self, original):
        super().__init__(str(original))
        self.error_type = type(original).__name__
        self.original_traceback = ''.join(traceback.format_exception(type(original), original, original.__traceback__))


def missing_grasp_target(exc):
    if type(exc) is not AssertionError or str(exc) != 'target_pose cannot be None for move action.':
        return False
    frames = traceback.extract_tb(exc.__traceback__)
    return (any(Path(f.filename).name == '_base_task.py' and f.name == 'grasp_actor' for f in frames)
            and bool(frames) and Path(frames[-1].filename).as_posix().endswith('/envs/utils/action.py')
            and frames[-1].name == '__init__')


def run_expert_plan(task):
    try:
        return task.play_once()
    except AssertionError as exc:
        if not missing_grasp_target(exc):
            raise
        raise ExpertPlanningFailure(exc) from exc


def planning_failure_record(exc):
    return dict(eligible=False, reason='expert_missing_grasp_target', error_type=exc.error_type,
                error=str(exc), traceback=exc.original_traceback)
