"""Explicit privileged teacher entry contract for a verified held actor."""
def enable_held_continuation(task,actor_name,side,on_entry):
    if side not in ('left','right'):raise ValueError('Invalid verified owner arm')
    used=False
    def grasp_actor(actor,arm_tag,pre_grasp_dis=.1,grasp_dis=0,gripper_pos=0.,contact_point_id=None):
        nonlocal used
        if used:raise RuntimeError('Held continuation contract may only replace one entry')
        if actor.get_name()!=actor_name or str(arm_tag)!=side:raise RuntimeError('Unverified actor or owner arm')
        used=True
        on_entry(dict(actor=actor_name,side=side,skipped='pregrasp and approach',retained='close_gripper',close_target=float(gripper_pos),native_count=int(task.take_action_cnt)))
        return task.close_gripper(arm_tag=arm_tag,pos=gripper_pos)
    task.grasp_actor=grasp_actor
