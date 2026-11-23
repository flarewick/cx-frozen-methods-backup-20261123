"""Fixed A-head preservation-source contrast, with approved native successes only."""
import argparse
from collections import Counter
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .audit import digest
from .paths import ROOT, new_run, output_path, write_json, replace_json
from .native_history_cache import read, arrays, require, verify
from .train_output_support import BASE, original_load, model_from_initial, compare_control, loss
from .train_structured_action import tensors
from .structured_action_adapter import FORWARD_KEYS

NATIVE = ROOT/'outputs/feature_caches/native_history_20261005_030329_014198'
REFERENCE = ROOT/'outputs/effect_learning/output_support_20261005_082307_793921'
DESIGN = ROOT/'outputs/reports/attribution/output_support_accepted_20261005_123653_284756/next_experiments.json'
CONDITIONS = ('original_control', 'native_preservation')


def approve_rows(p):
    train = [i for i,r in enumerate(p['rows']) if r['target_kind']=='success_preservation' and r['split']=='train_candidate']
    held = [i for i,r in enumerate(p['rows']) if r['target_kind']=='success_preservation' and r['split']=='development_preservation_heldout']
    require((len(train),len(held))==(40,8),'Frozen native selection changed')
    for i in train+held:
        r=p['rows'][i]
        require(r['source_success'] is True and r['teacher_entry'] is None,'Unqualified teacher/failure label')
        require(r['executed']==sum(r['target_tail_mask']) and 0<r['executed']<=32,'Actual valid tail')
        require(r['previous_query_native_action_gap'] in (None,32),'Native history time')
    scenes=lambda ids:{(p['rows'][i]['task'],p['rows'][i]['seed']) for i in ids}
    require(len(scenes(train))==5 and len(scenes(held))==2 and not scenes(train)&scenes(held),'Scene split')
    return train,held


def prepare():
    design=read(DESIGN);verify(design['source_sha256']);n=read(NATIVE/'protocol.json');na=read(NATIVE/'audit.json')
    require(na['passed'] and not n['training_allowed'],'Keep original candidate cache immutable')
    verify(na['source_sha256']);verify(n['source_sha256']);p,old,x,y=original_load(BASE)
    tr,he=approve_rows(n);raw=arrays(NATIVE/'inputs.npz');recorded=arrays(NATIVE/'recorded_bases.npz')
    require(set(raw)==FORWARD_KEYS,'Legal9 inputs only')
    weights=np.zeros(40,np.float32);counts=Counter((n['rows'][i]['task'],n['rows'][i]['seed']) for i in tr)
    for j,i in enumerate(tr):weights[j]=.5/(len(counts)*counts[n['rows'][i]['task'],n['rows'][i]['seed']])
    require(abs(float(weights.sum())-.5)<1e-6,'Native preservation mass')
    for i in tr+he:
        row=n['rows'][i];base=arrays(row['base'])['base_action']
        require(base.tobytes()==recorded['base'][i].tobytes(),'Actual base mismatch')
        require(recorded['source_valid'][i].tolist()==row['target_tail_mask'],'Native valid mask mismatch')
    background_scenes=set()
    for r in p['rows']:
        parts=r['group'].split(':')
        require(len(parts)>=2 and parts[0]==r['task'] and parts[1].isdigit(),'Background scene identity schema')
        background_scenes.add((r['task'],int(parts[1])))
    require(not background_scenes & {(n['rows'][i]['task'],n['rows'][i]['seed']) for i in he},'Original background leaks held scene')
    anchor=np.asarray([r['kind']=='anchor' for r in p['rows']]);require(int(anchor.sum())==10,'Original anchors')
    modified=y['row_weights'].copy();modified[anchor]*=.5
    require(abs(float(y['row_weights'].sum())-4)<1e-6 and abs(float(modified.sum()+weights.sum())-4)<1e-6,'Fixed total mass4')
    run=new_run('native_preservation',group=('effect_learning',))
    for name,ids in (('train',tr),('held',he)):
        np.savez_compressed(run/(name+'_inputs.npz'),**{k:v[ids].copy() for k,v in raw.items()})
        np.savez_compressed(run/(name+'_labels.npz'),base=recorded['base'][ids].copy(),target=recorded['base'][ids].copy(),
            valid=recorded['source_valid'][ids].copy(),row_weights=weights if name=='train' else np.full(8,1/8,np.float32))
    np.savez_compressed(run/'background_weights.npz',row_weights=modified)
    paths=[DESIGN,NATIVE/'protocol.json',NATIVE/'audit.json',NATIVE/'inputs.npz',NATIVE/'recorded_bases.npz',
        REFERENCE/'training_protocol.json',BASE/'inputs.npz',BASE/'labels.npz',BASE/'initial_seed0.pt',Path(__file__),
        ROOT/'scripts/train_native_preservation.sh']
    paths += list(run.glob('*.npz'))
    q=read(REFERENCE/'training_protocol.json')
    protocol=dict(stage='native_preservation_recipe',status='frozen_before_training',conditions=list(CONDITIONS),
        parent=str(BASE),reference=str(REFERENCE),native_cache=str(NATIVE),native_train_indices=tr,native_held_indices=he,
        train_scenes=[list(t) for t in sorted(counts)],held_scenes=[list(t) for t in sorted({(n['rows'][i]['task'],n['rows'][i]['seed']) for i in he})],
        fixed_reference_protocol=q,seed=0,updates=1400,lr=.0003,weight_decay=.01,parameters=42934,
        old_masses=dict(expert=1,anchor=1,recovery=2),new_masses=dict(expert=1,old_anchor=.5,native_anchor=.5,recovery=2),
        actor_inputs='9 legal fields+actual base; no task/seed/outcome/teacher phase',
        targets='Native zero residual only on actual executed steps, not optimal-action or local-benefit labels',
        training_approved_success_preservation_only=True,unqualified_teacher_rows_excluded=11,
        background_rows=611,native_train_rows=40,native_held_rows=8,total_mass=4,
        history='Unchanged original611; new native past is same-source preceding32 executed actions',
        time_limit_s=600,source_sha256={str(f):digest(f) for f in paths})
    write_json(run/'protocol.json',protocol);print('RUN',run,flush=True);return run


def load(run):
    q=read(run/'protocol.json');verify(q['source_sha256']);_,_,x,y=original_load(BASE)
    return q,x,y,arrays(run/'train_inputs.npz'),arrays(run/'train_labels.npz'),arrays(run/'held_inputs.npz'),arrays(run/'held_labels.npz')


def recipe_loss(model,x,y,nx,ny,condition,weights):
    if condition=='original_control':return loss(model,x,y)
    require(condition=='native_preservation','Unknown recipe')
    altered=dict(y,row_weights=weights)
    return loss(model,x,altered)+loss(model,nx,ny)


def preflight(run):
    q,x,y,nx,ny,hx,hy=load(run);model=model_from_initial(q['fixed_reference_protocol'],'A_bounded_clipped')
    require(sum(p.numel() for p in model.parameters())==42934,'Shared allocation')
    require(np.array_equal(ny['base'],ny['target']) and np.array_equal(hy['base'],hy['target']),'Zero preservation labels')
    model.load_state_dict(torch.load(REFERENCE/'A_bounded_clipped_seed0.pt',map_location='cpu',weights_only=True)['model'])
    xt,yt,nxt,nyt=tensors(x),tensors(y),tensors(nx),tensors(ny)
    control=recipe_loss(model,xt,yt,nxt,nyt,'original_control',torch.from_numpy(arrays(run/'background_weights.npz')['row_weights']))
    require(torch.equal(control,loss(model,xt,yt)),'Control loss changed')
    gradients={};norms={}
    rows=read(BASE/'protocol.json')['rows']
    for group in ('expert','anchor','recovery','native'):
        if group=='native':value=loss(model,nxt,nyt)
        else:
            w=yt['row_weights'].clone();w[torch.tensor([r['kind']!=group for r in rows])]=0
            value=loss(model,xt,dict(yt,row_weights=w))
        g=torch.autograd.grad(value,tuple(model.parameters()),allow_unused=True)
        vector=torch.cat([v.flatten() for v in g if v is not None]).double();gradients[group]=vector
        norms[group]=dict(norm=float(vector.norm()),weighted_loss=float(value.detach()))
    cos={a+'__'+b:float(torch.dot(gradients[a],gradients[b])/(gradients[a].norm()*gradients[b].norm()))
        for a,b in (('native','recovery'),('native','expert'),('native','anchor'))}
    write_json(run/'preflight.json',dict(passed=True,GPU_started=False,original_control_loss_equal=True,
        total_mass_verified=4,approved_rows=40,excluded_unqualified_rows=11,held_whole_scene_rows=8,
        gradient_groups=norms,gradient_cosine=cos,gradient_scope='At frozen originalA; numeric conflict clue, not causal failure or benefit label'))
    print('PREFLIGHT_PASS',cos,flush=True)


def train(run):
    q,x,y,nx,ny,hx,hy=load(run);require(read(run/'preflight.json')['passed'],'CPU gate')
    require(torch.cuda.is_available() and not (run/'training_environment.json').exists(),'Fresh CUDA job required')
    write_json(run/'training_environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),backbone_loaded=False))
    xt,yt,nxt,nyt=tensors(x,'cuda'),tensors(y,'cuda'),tensors(nx,'cuda'),tensors(ny,'cuda')
    weights=torch.from_numpy(arrays(run/'background_weights.npz')['row_weights']).cuda();began=time.monotonic();reports=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Native recipe600s bound')));signal.alarm(600)
    try:
        for condition in CONDITIONS:
            model=model_from_initial(q['fixed_reference_protocol'],'A_bounded_clipped','cuda')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);history=[]
            for step in range(1,1401):
                optimizer.zero_grad(set_to_none=True);value=recipe_loss(model,xt,yt,nxt,nyt,condition,weights)
                require(bool(torch.isfinite(value)),'Finite loss');value.backward();optimizer.step()
                if step==1 or step%200==0:
                    item=dict(step=step,loss=float(value.detach()));history.append(item)
                    replace_json(run/'progress.json',dict(condition=condition,**item));print(condition,item,flush=True)
            if condition=='original_control':
                reproduced=compare_control(model.state_dict());write_json(run/'control_reproduction.json',reproduced)
                require(reproduced['exact'],'Original control must reproduce prior A')
            path=run/(condition+'_seed0.pt');torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                metadata=dict(condition=condition,protocol_sha256=digest(run/'protocol.json'),updates=1400,seed=0)),path)
            reports.append(dict(condition=condition,checkpoint=str(path),checkpoint_sha256=digest(path),history=history))
        write_json(run/'training_result.json',dict(status='complete',reports=reports,elapsed_s=time.monotonic()-began,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),backbone_loaded=False))
    except BaseException:
        write_json(run/'training_failure.json',dict(traceback=traceback.format_exc(),completed_models=len(reports)));raise
    finally:signal.alarm(0)
    print('TRAIN_COMPLETE',run,flush=True)


def audit(run):
    q,x,y,nx,ny,hx,hy=load(run);result=read(run/'training_result.json');reports=[];predictions={}
    rows=read(BASE/'protocol.json')['rows']
    for item in result['reports']:
        require(digest(Path(item['checkpoint']))==item['checkpoint_sha256'],'Checkpoint identity')
        model=model_from_initial(q['fixed_reference_protocol'],'A_bounded_clipped');model.load_state_dict(torch.load(item['checkpoint'],map_location='cpu',weights_only=True)['model']);model.eval()
        stats={}
        with torch.inference_mode():
            for name,xx,yy in (('background',x,y),('native_train',nx,ny),('native_held',hx,hy)):
                out=model(tensors(xx),torch.from_numpy(yy['base']));delta=out['delta'].numpy()
                predictions[item['condition'],name]=delta.copy()
                stats[name]=dict(normalized_loss=float(loss(model,tensors(xx),tensors(yy))),
                    applied_delta_rms=float(np.sqrt(np.mean(delta**2))),max_abs=float(np.max(np.abs(delta))))
            group_losses={}
            for task,kind in sorted({(r['task'],r['kind']) for r in rows}):
                w=y['row_weights'].copy();w[np.asarray([r['task']!=task or r['kind']!=kind for r in rows])]=0
                group_losses[task+'/'+kind]=float(loss(model,tensors(x),tensors(dict(y,row_weights=w))))
        reports.append(dict(condition=item['condition'],sets=stats,original_group_losses=group_losses))
    original,new=reports;held_ratio=new['sets']['native_held']['normalized_loss']/original['sets']['native_held']['normalized_loss']
    recovery_ratios={k:new['original_group_losses'][k]/v for k,v in original['original_group_losses'].items() if k.endswith('/recovery')}
    proposal=dict(native_held_improved=held_ratio<1,held_loss_ratio=held_ratio,recovery_loss_ratios=recovery_ratios,
        native_train_loss_ratio=new['sets']['native_train']['normalized_loss']/original['sets']['native_train']['normalized_loss'],
        physical_success_improvement=False,automatic_promotion=False,
        rule='Report held preservation and recovery separately; decide physical test before choosing any threshold, not from training loss.')
    write_json(run/'audit.json',dict(passed=True,GPU_started=False,reports=reports,diagnostics=proposal,
        original_control=read(run/'control_reproduction.json'),source_sha256={str(run/'training_result.json'):digest(run/'training_result.json')}))
    print('AUDIT',proposal,flush=True)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('phase',choices=('prepare','preflight','train','audit'));ap.add_argument('--run',type=Path)
    args=ap.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.phase=='prepare':prepare();return
    require(args.run is not None,'Run required');{'preflight':preflight,'train':train,'audit':audit}[args.phase](output_path(args.run))


if __name__=='__main__':main()
