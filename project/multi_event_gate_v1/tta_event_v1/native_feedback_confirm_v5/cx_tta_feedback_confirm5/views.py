"""Three useful controlled views of the same actual observation, one anchor."""
import numpy as np

VIEWS=('actual','image_only','state_only')

def view_inputs(images,state,initial_images,initial_state):
    if len(images)!=3 or len(initial_images)!=3:raise ValueError('Three cameras required')
    for group in (images,initial_images):
        if any(np.asarray(v).ndim!=3 or v.shape[-1]!=3 or v.dtype!=np.uint8 for v in group):raise ValueError('Actual RGB arrays required')
    if np.asarray(state).shape!=(14,) or np.asarray(initial_state).shape!=(14,) or not np.isfinite(state).all() or not np.isfinite(initial_state).all():
        raise ValueError('Finite current/fork command states required')
    # New arrays prevent the official transforms from mutating shared observations.
    return [(tuple(v.copy() for v in rgb),s.copy()) for rgb,s in
            ((images,state),(images,initial_state),(initial_images,state))]

def decompose(visual,anchor):
    visual=np.asarray(visual,np.float64);anchor=np.asarray(anchor,np.float64)
    if visual.shape[-5:]!=(3,3,2,2,32) or anchor.shape!=(3,2,2,32):raise ValueError('View/camera/partition dimensions differ')
    actual=visual[...,0,:,:,:,:]-anchor
    image=visual[...,1,:,:,:,:]-anchor
    state=visual[...,2,:,:,:,:]-anchor
    return dict(actual=actual,image_only=image,state_only=state,interaction=actual-image-state)

def summarize_commands(commands,dt,state):
    from cx_tta_response.data import command_summary
    return command_summary(commands,dt,state)

def batched_encode(policy,views,prompt):
    """Same frozen prefix computation, batch3; actual numerical parity needs GPU."""
    import jax
    import torch
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from cx_tta.cache_joint_prefix import digest_array
    inputs=[]
    for images,state in views:
        obs=dict(state=state.copy(),prompt=prompt,images={k:im.transpose(2,0,1).copy() for k,im in zip(('cam_high','cam_right_wrist','cam_left_wrist'),images)})
        inputs.append(policy._input_transform(obs))
    tensors=jax.tree.map(lambda *parts:torch.from_numpy(np.stack(parts).copy()).to('cuda'),*inputs)
    observation=Observation.from_dict(tensors);m=policy._model
    with torch.inference_mode():
        imgs,im_masks,tokens,lang_mask,_=m._preprocess_observation(observation,train=False)
        embeddings,pad,att=m.embed_prefix(imgs,im_masks,tokens,lang_mask)
        mask=m._prepare_attention_masks_4d(make_att_2d_masks(pad,att));pos=torch.cumsum(pad,dim=1)-1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
        (hidden,_),_=m.paligemma_with_expert.forward(attention_mask=mask,position_ids=pos,past_key_values=None,inputs_embeds=[embeddings,None],use_cache=False)
        hidden=hidden.float();features=[]
        for i in range(len(views)):
            spatial=hidden[i,:768].reshape(3,16,16,2048).reshape(3,2,8,2,8,2048).mean(dim=(2,4)).reshape(-1)
            text=hidden[i,768:][lang_mask[i]]
            features.append(torch.cat([spatial,text.mean(0),text[-1]]).cpu().numpy().astype(np.float16))
    result=np.stack(features)
    if result.shape!=(3,28672) or not np.isfinite(result).all():raise ValueError('Wrong controlled prefix output')
    return result,dict(token_sha256=[digest_array(v.cpu().numpy()) for v in tokens],valid_language_tokens=[int(v.sum()) for v in lang_mask])
