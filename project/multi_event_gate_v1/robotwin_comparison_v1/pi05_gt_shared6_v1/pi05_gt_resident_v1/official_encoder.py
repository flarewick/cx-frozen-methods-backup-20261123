"""Original G singleton auxiliary prefix; restore attention configuration before base inference."""
import numpy as np
from cx_tta.cache_joint_prefix import encode

def encode_actual(policy,views,prompt):
 images,state=views[0]
 assert len(views)==3 and all(np.array_equal(state,s)and all(np.array_equal(a,b)for a,b in zip(images,ims,strict=True))for ims,s in views)
 config=policy._model.paligemma_with_expert.paligemma.language_model.config;implementation=config._attn_implementation
 try:feature,meta=encode(policy,images,state,prompt)
 finally:config._attn_implementation=implementation
 return np.repeat(feature[None],3,axis=0),dict(token_sha256=[meta['token_sha256']]*3,valid_language_tokens=[meta['valid_language_tokens']]*3,original_G_singleton_prefix=True,attention_configuration_restored=True,identical_control_views_reused=True)
