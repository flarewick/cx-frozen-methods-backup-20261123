"""Read-only instantaneous contact labels, not contact integrated over actions."""
import numpy as np
from .short_window_contract import capture_frame as capture_observation

def capture_frame(task,observation,run,branch,offset,mapping,scene_ids):
    frame=capture_observation(task,observation,run,branch,offset,mapping,scene_ids)
    count=task.take_action_cnt;ticks=getattr(task,'_cx_intervention_ticks',0)
    contacts=[]
    for contact in task.scene.get_contacts():
        ids=[int(body.entity.per_scene_id) for body in contact.bodies]
        if not any(i in entity_ids for i in ids for entity_ids in mapping.values()):continue
        points=[]
        for point in contact.points:
            position=np.asarray(point.position,float)
            if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid contact position')
            points.append(position.tolist())
        contacts.append(dict(entity_ids=ids,entity_names=[body.entity.name for body in contact.bodies],positions_world_m=points))
    if task.take_action_cnt!=count or getattr(task,'_cx_intervention_ticks',0)!=ticks:
        raise ValueError('Contact recorder advanced physics')
    frame['contacts_offline_only']={'scope':'instantaneous snapshot; absence does not imply no contact during window','pairs':contacts}
    return frame
