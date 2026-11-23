"""Fresh seed17 original architecture plus deterministic action-visible increment."""
import torch
from native_direction_training_v2.train import model_for as original_factory
from native_action_visible_response_v1.model import from_legacy,ActionVisibleResponse

def model_for(auth):
 original,config,kernel=original_factory(auth);model=from_legacy(original)
 assert type(model)is ActionVisibleResponse and int(model.response.ARM_visible_interface_version)==1
 assert all(torch.equal(a,b)for a,b in zip(model.response.parameters(),original.response.parameters(),strict=True))
 return model,config,kernel
