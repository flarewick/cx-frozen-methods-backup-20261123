"""A diagnostic readout only: no actions, simulator, router, or TTA inputs."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .paths import write_json
from .audit import digest
from .probe_candidates import read,verify
from .structured_features import load_structured
from .diagnose_teacher_observability import pose7,balanced,transform_blocks
from .diagnose_teacher_invariants import interpolate
from .robot_kinematics import RobotFK
from .audit_pose_capacity import numpy_fk
from .train_rotation_curriculum import state_hash

MODES=('uniform','conditional','visual_disabled')
SEEDS=(0,1,2)


class StructuredProbe(nn.Module):
    def __init__(self,channels,reg_mean,phase_prior):
        super().__init__();self.channels=channels
        self.query=nn.Linear(channels*2+28,16);self.key=nn.Linear(channels,16,bias=False)
        self.position=nn.Parameter(torch.randn(3,4,16)*.01)
        dim=2*14*channels+28+36+1
        self.trunk=nn.Sequential(nn.Linear(dim,32),nn.SiLU(),nn.Linear(32,64),nn.SiLU())
        self.head=nn.Linear(64,11);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.query.weight);nn.init.zeros_(self.query.bias)
        self.register_buffer('reg_mean',torch.as_tensor(reg_mean,dtype=torch.float32))
        self.register_buffer('phase_prior',torch.as_tensor(phase_prior,dtype=torch.float32))

    def attention(self,x,which,mode):
        n=len(x['state']);visual=x[which+'visual'].reshape(n,3,4,self.channels)
        if mode=='conditional':
            q=self.query(torch.cat([x['text'].flatten(1),x['state'],x['past_state']],1))
            k=self.key(visual)+self.position
            logits=(k*q[:,None,None]).sum(-1)/4
            weights=torch.softmax(logits,dim=-1)*4
        else:weights=torch.ones(n,3,4,device=visual.device,dtype=visual.dtype)
        values=visual*weights[...,None]
        if mode=='visual_disabled':values=torch.zeros_like(values)
        return values.flatten(1),weights

    def forward(self,x,mode):
        if mode not in MODES:raise ValueError('Unknown fixed probe condition')
        cur,cw=self.attention(x,'',mode);past,pw=self.attention(x,'past_',mode)
        merged=torch.cat([cur,x['text'].flatten(1),past,x['past_text'].flatten(1),
            x['state'],x['past_state'],x['ee_pose'].flatten(1),x['past_ee_pose'].flatten(1),x['has_history']],1)
        output=self.head(self.trunk(merged))
        return dict(regression=output[:,:7]+self.reg_mean,phase=output[:,7:]+self.phase_prior,
            current_attention=cw,past_attention=pw)


def probe_loss(output,reg,phase,weights):
    loss=(output['regression']-reg).square().mean(1)
    loss+=nn.functional.cross_entropy(output['phase'],phase,reduction='none')
    return (loss*weights).sum()


def metrics(pred,phase_logits,reg,phase,weights):
    normal=pred[:,3:6];truth=reg[:,3:6];norm=np.linalg.norm(normal,axis=1)
    angle=np.degrees(np.arccos(np.clip(np.sum(normal*truth,axis=1)/np.maximum(norm,1e-8),-1,1)))
    angle[norm<1e-8]=180
    return dict(position_cm=float(weights@np.linalg.norm(pred[:,:3]-reg[:,:3],axis=1)*10),
        normal_angle_deg=float(weights@angle),gripper_mae=float(weights@np.abs(pred[:,6]-reg[:,6])*.2),
        phase_accuracy=float(weights@(np.argmax(phase_logits,axis=1)==phase)),invalid_normals=int((norm<1e-8).sum()))


def prepare_probes(run):
    p,all_inputs=load_structured(run);cp=read(Path(p['cache'])/'protocol.json')
    diag=Path(p['diagnostic']);inv=read(diag/'invariants.json');dr=read(diag/'result.json')
    assert read(diag/'audit.json')['passed']
    ii=np.array([i for i,r in enumerate(p['rows']) if r['task']=='turn_switch']);rows=[p['rows'][i] for i in ii]
    inputs={k:v[ii] for k,v in all_inputs.items()}
    with np.load(Path(p['cache'])/'features.npz') as z:states=z['state'][np.array(cp['feature_current_indices'])[ii]]
    with np.load(Path(p['cache'])/'data.npz') as z:
        target=z['target'][ii];valid=z['valid'][ii];phase=z['plan_ids'][ii,0].astype(np.int64)
    old=read(Path(p['paired'])/'protocol.json')
    robots=[RobotFK(read(Path(old['config'])),side) for side in ('left','right')]
    ee=numpy_fk(robots[1],states[:,7:13].astype(float))
    contacts=[];normals=[];grips=[]
    for j,row in enumerate(rows):
        contacts.append(next(x['contact_xyz'] for x in inv['functional_points'] if x['source']==row['source'] and x['plan']==1))
        branch=next(x for x in dr['branches'] if x['run']==row['source'])
        normals.append(-pose7(branch['switch_root_pose'])[:3,0])
        grips.append(float(target[j,:,13][valid[j]].mean()))
    contacts=np.array(contacts);normals=np.array(normals)
    relp=np.einsum('ni,nij->nj',contacts-ee[:,:3,3],ee[:,:3,:3]);reln=np.einsum('ni,nij->nj',normals,ee[:,:3,:3])
    reg=np.c_[relp/.1,reln,np.array(grips)/.2].astype(np.float32)
    assert len(rows)==20 and set(phase).issubset({0,1,2})
    np.savez_compressed(run/'probe_labels.npz',regression=reg,phase=phase,contacts_world=contacts,normals_world=normals,ee=ee,indices=ii)
    with np.load(Path(p['paired'])/'data.npz') as z:old_ctx=z['context'][np.array(old['new_indices'])[ii]].copy()
    jobs=[];initials=[];baselines=[]
    for scene in sorted({r['scene_group'] for r in rows}):
        train=np.array([i for i,r in enumerate(rows) if r['scene_group']!=scene]);held=np.array([i for i,r in enumerate(rows) if r['scene_group']==scene])
        tw=balanced(rows,train);hw=balanced(rows,held);mean=tw@reg[train]
        counts=np.array([tw[phase[train]==c].sum() for c in range(4)])+.01;prior=np.log(counts/counts.sum())
        job=dict(scene=scene,train=train.tolist(),held=held.tolist(),train_weights=tw.tolist(),held_weights=hw.tolist(),mean=mean.tolist(),phase_prior=prior.tolist())
        jobs.append(job)
        # Strong training-only world-goal prior, then transform into each held wrist frame.
        basep=(tw@contacts[train])-ee[:,:3,3];basep=np.einsum('ni,nij->nj',basep,ee[:,:3,:3])
        basen=np.tile(tw@normals[train],(len(rows),1));basen/=np.linalg.norm(basen,axis=1)[:,None]
        basen=np.einsum('ni,nij->nj',basen,ee[:,:3,:3])
        base=np.c_[basep/.1,basen,np.repeat(mean[-1],len(rows))]
        baselines.append(dict(scene=scene,method='training_world_mean',held=metrics(base[held],np.tile(prior,(len(held),1)),reg[held],phase[held],hw)))
        # Same canonical labels and legal command-FK input for both analytic controls.
        blocks={'old_pca_with_fk':[old_ctx,inputs['ee_pose'].reshape(20,-1),inputs['past_ee_pose'].reshape(20,-1)],
            'structured_flat_with_fk':[inputs['visual'].reshape(20,-1),inputs['text'].reshape(20,-1),
                inputs['past_visual'].reshape(20,-1),inputs['past_text'].reshape(20,-1),inputs['state'],inputs['past_state'],
                inputs['ee_pose'].reshape(20,-1),inputs['past_ee_pose'].reshape(20,-1)]}
        for name,bb in blocks.items():
            x=transform_blocks(bb,train);prediction=interpolate(x,reg,train,tw);logits=interpolate(x,np.eye(4)[phase],train,tw)
            baselines.append(dict(scene=scene,method=name,train=metrics(prediction[train],logits[train],reg[train],phase[train],tw),
                held=metrics(prediction[held],logits[held],reg[held],phase[held],hw)))
        for seed in SEEDS:
            torch.manual_seed(seed);m=StructuredProbe(p['channel_dim'],mean,prior)
            file=run/f'probe_initial_{scene.split(":")[-1]}_s{seed}.pt';torch.save(m.state_dict(),file)
            initials.append(dict(scene=scene,seed=seed,path=str(file),sha256=digest(file),state_sha256=state_hash(m)))
    files=[run/'protocol.json',run/'interface_audit.json',run/'probe_labels.npz',diag/'result.json',diag/'invariants.json',diag/'audit.json',
        Path(__file__),Path(__file__).with_name('train_structured_probe.py'),Path(__file__).with_name('audit_structured_probe.py'),
        Path(__file__).parents[1]/'scripts/train_structured_probe.sh',*[Path(i['path']) for i in initials]]
    protocol=dict(status='frozen_before_training',rows=rows,indices=ii.tolist(),jobs=jobs,initializations=initials,
        modes=list(MODES),seeds=list(SEEDS),updates=600,lr=.0003,weight_decay=.01,parameters=sum(v.numel() for v in m.parameters()),
        development_criterion='At least2of3scenes: in all3paired seeds conditional position improves over uniform and world-mean baseline; normal worsens by no more than1degree vsuniform. Signal only, not deployment approval.',
        baselines=baselines,source_sha256={str(f):digest(f) for f in files},
        loss='Equal regression and stage terms; regression mean of relative-position/.1, unit-face-normal, future-valid-grip-mean/.2 squared errors. '
            'Same scene/branch/row weights in every condition. Diagnostics only, teacher stage is never an input.',
        distinction='Only learned per-camera spatial weighting enabled vs uniform, identical observable inputs/common model/init/update budget. '
            'visual_disabled additionally removes both current and past visual values as information control; language/proprio/FK retained.',
        input_normalization='Expert-only shared channel PCA32; raw arm-state/pi and fixed command-FK position/.1. No recovery-fitted input statistics.',
        evaluation='Final600, three paired seeds, each whole scene. Include train fit, held geometry/stage/gripper errors and analytic controls. '
            'All results retained;3scenes only. Does not train an action adapter or establish task success or full closed-loop safety.')
    write_json(run/'probe_protocol.json',protocol);print('PROBES_PREPARED',run,flush=True)


def load_probes(run):
    p,all_inputs=load_structured(run);q=read(run/'probe_protocol.json');verify(q)
    idx=q['indices'];inputs={k:v[idx].copy() for k,v in all_inputs.items()}
    with np.load(run/'probe_labels.npz') as z:labels={k:z[k].copy() for k in z.files}
    return p,q,inputs,labels
