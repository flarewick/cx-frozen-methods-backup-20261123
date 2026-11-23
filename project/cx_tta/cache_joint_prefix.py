"""Frozen current RGB/instruction/state VLM prefix; no action sampling or simulator."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from .audit import digest
from .paths import ROOT, new_run, write_json, replace_json


def encode(policy, images, state, prompt):
    import jax
    import torch
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    obs={'state':state.copy(),'prompt':prompt,'images':{k:im.transpose(2,0,1).copy() for k,im in zip(('cam_high','cam_right_wrist','cam_left_wrist'),images)}}
    inputs=policy._input_transform(obs)
    inputs=jax.tree.map(lambda x:torch.from_numpy(np.asarray(x).copy()).to('cuda')[None],inputs)
    observation=Observation.from_dict(inputs)
    m=policy._model
    with torch.inference_mode():
        imgs,im_masks,tokens,lang_mask,_=m._preprocess_observation(observation,train=False)
        embeddings,pad,att=m.embed_prefix(imgs,im_masks,tokens,lang_mask)
        mask=m._prepare_attention_masks_4d(make_att_2d_masks(pad,att))
        pos=torch.cumsum(pad,dim=1)-1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
        (hidden,_),_=m.paligemma_with_expert.forward(attention_mask=mask,position_ids=pos,past_key_values=None,inputs_embeds=[embeddings,None],use_cache=False)
        hidden=hidden[0].float();length=3*256
        if hidden.shape[0]!=length+lang_mask.shape[1] or hidden.shape[1]!=2048:raise ValueError('Unexpected prefix geometry')
        spatial=hidden[:length].reshape(3,16,16,2048).reshape(3,2,8,2,8,2048).mean(dim=(2,4)).reshape(-1)
        text=hidden[length:][lang_mask[0]]
        feature=torch.cat([spatial,text.mean(0),text[-1]]).cpu().numpy()
    if feature.shape!=(28672,) or not np.isfinite(feature).all():raise ValueError('Invalid joint feature')
    return feature.astype(np.float16),dict(image_masks=[bool(x.item()) for x in im_masks],valid_language_tokens=int(lang_mask.sum()),token_sha256=digest_array(tokens.cpu().numpy()),feature_dimension=len(feature))


def digest_array(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    if not a.gpu.isdigit():p.error('Numeric GPU required')
    a.plan=a.plan.resolve()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['JAX_PLATFORMS']='cpu'
    import torch
    import h5py
    from .cache_expert import create_base_policy,CHECKPOINT
    from .dataset import read_observation
    from .evaluate_b2 import configure_runtime,resolve_tokenizer
    plan=json.loads(a.plan.read_text());records=plan['records']
    if plan['status']!='frozen':raise ValueError('Frozen pilot required')
    run=new_run('joint_prefix_smoke' if a.smoke_only else 'joint_prefix',group=('feature_caches',));print(run,flush=True)
    runtime=run/'runtime';runtime.mkdir();configure_runtime(run,runtime,resolve_tokenizer())
    torch.set_num_threads(2);torch.manual_seed(0);torch._dynamo.config.disable=True
    if not torch.cuda.is_available():raise RuntimeError('GPU access unavailable')
    policy=create_base_policy();policy._model.eval().requires_grad_(False)
    manifest=dict(plan=str(a.plan.resolve()),plan_sha256=digest(a.plan),code_sha256=digest(Path(__file__)),checkpoint=str(CHECKPOINT),smoke_only=a.smoke_only,feature='Frozen last prefix hidden state: 3 camera 2x2 pooled + valid text mean + last valid text; 28672 float16. Eager auxiliary extraction; base actions unchanged and never resampled.',gpu=torch.cuda.get_device_name(0),torch=torch.__version__,trainable_backbone_parameters=sum(p.numel() for p in policy._model.parameters() if p.requires_grad))
    write_json(run/'manifest.json',manifest);results=[];probe=[];all_seconds=[]
    chosen=records
    if a.smoke_only:
        chosen=[next(r for r in records if r['kind']=='expert' and r['task']==t and r['pilot_split']=='fit') for t in sorted({r['task'] for r in records})]
    for rid,row in enumerate(chosen):
        if row['kind']=='expert':
            if digest(Path(row['cache']))!=row['cache_sha256'] or digest(Path(row['source']))!=row['source_sha256']:raise ValueError('Changed expert input')
            with np.load(row['cache']) as z:frames=z['frame_index']
            if a.smoke_only:frames=frames[len(frames)//2:len(frames)//2+1]
            handle=h5py.File(row['source'],'r')
            samples=[(int(f),None) for f in frames]
        else:
            if digest(Path(row['checkpoint']))!=row['checkpoint_sha256']:raise ValueError('Changed anchor checkpoint')
            with np.load(row['checkpoint']) as z:stored=dict(z)
            samples=[(row['chunk'],stored)];handle=None
        features=[];metas=[]
        try:
            for frame,stored in samples:
                if stored is None:images,state=read_observation(handle,frame)
                else:images=[stored[k+'_rgb'] for k in ('head','right','left')];state=stored['command_state']
                before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state().clone();start=time.perf_counter()
                feature,meta=encode(policy,images,state,row['instruction']);elapsed=time.perf_counter()-start
                if not torch.equal(before_cpu,torch.get_rng_state()) or not torch.equal(before_gpu,torch.cuda.get_rng_state()):raise ValueError('Feature extraction consumed policy RNG')
                if a.smoke_only:
                    repeated,_=encode(policy,images,state,row['instruction']);blank,bmeta=encode(policy,images,state,'')
                    if not np.array_equal(feature,repeated) or np.array_equal(feature,blank):raise ValueError('Repeatability/instruction sensitivity failed')
                    probe.append(dict(task=row['task'],frame=frame,repeat_exact=True,instruction_change_l2=float(np.linalg.norm(feature.astype(float)-blank.astype(float))),instruction_tokens_changed=meta['token_sha256']!=bmeta['token_sha256'],rng_unchanged=True,seconds=elapsed))
                features.append(feature);metas.append(dict(frame=frame,**meta));all_seconds.append(elapsed)
        finally:
            if handle is not None:handle.close()
        dest=run/f'features_{rid:04d}.npz';np.savez_compressed(dest,joint=np.asarray(features),frames=np.asarray([m['frame'] for m in metas]))
        results.append(dict(source=row,file=str(dest),sha256=digest(dest),samples=metas))
        replace_json(run/'progress.json',dict(records=len(results),total_records=len(chosen),observations=len(all_seconds),last_seconds=all_seconds[-1]));print('CACHED',len(results),'/',len(chosen),'observations',len(all_seconds),flush=True)
    write_json(run/'result.json',dict(status='complete',records=results,smoke=probe,observations=len(all_seconds),median_feature_seconds=float(np.median(all_seconds)),scope='Read existing current observations only. No teacher actions, outcomes, GT geometry or task IDs supplied to encoder; successful episode labels only selected anchor sources offline. Instruction sensitivity checks interface, NOT useful language grounding or task success.'))
    print('COMPLETE '+str(run),flush=True)

if __name__=='__main__':main()
