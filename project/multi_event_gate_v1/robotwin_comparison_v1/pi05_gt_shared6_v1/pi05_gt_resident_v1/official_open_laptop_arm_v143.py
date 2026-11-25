"""Restore the original expert-selected arm used by RoboTwin's success predicate."""


def restore(task, task_name):
    if task_name != 'open_laptop':
        return None

    from envs.utils import ArmTag, get_face_prod

    face_prod = get_face_prod(task.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
    arm_tag = ArmTag('left' if face_prod > 0 else 'right')
    previous = getattr(task, 'arm_tag', None)
    if previous is not None and previous != arm_tag:
        raise RuntimeError('Reconstructed open_laptop arm conflicts with RoboTwin play_once')
    task.arm_tag = arm_tag
    return str(arm_tag)
