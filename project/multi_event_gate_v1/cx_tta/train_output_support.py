"""Four fixed-budget S2o cells on the unchanged611-row S0 background."""
import argparse
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .audit import digest
from .paths import ROOT,new_run,output_path,artifact_path,write_json,replace_json
from .native_history_cache import BASE,arrays,read,verify,require
from .output_support_adapter import OutputSupportAdapter,CONDITIONS,support_loss,native_domain_stats
from .structured_action_adapter import StructuredActionAdapter
from .train_structured_action import tensors,state_hash,losses as old_losses
from .train_expert_history_action import parent_data
from .robot_kinematics import RobotFK
from .audit_local_geometry import joint_bounds

DESIGN=ROOT/'outputs/reports/protocols/output_label_support_20261005_075650_144185/design.json'
SCOPE='S2o matched output support and target clipping'


def original_load(_base):
    require(Path(_base)==BASE,'Fixed parent only')
    p,q,x,y,_,gaps=parent_data()
    verify({name:sha for name,sha in p['source_sha256'].items() if Path(name).is_file()})
    return p,q,x,y


def robot_limits(config):
    limits=np.zeros((14,2),np.float32)
    for off,side in ((0,'left'),(7,'right')):
        limits[off:off+6]=joint_bounds(RobotFK(config,side));limits[off+6]=[0,1]
    return limits


def prepare():
    design=read(DESIGN);verify(design['source_sha256'])
    p,q,x,y=original_load(BASE);require(len(y['base'])==611 and read(BASE/'training_audit.json')['passed'],'Audited611 background')
    require(list(design['conditions'])==list(CONDITIONS),'Sealed cell order')
    initial=torch.load(BASE/'initial_seed0.pt',map_location='cpu',weights_only=True)
    model=OutputSupportAdapter(32,q['geometry_mean'],list(CONDITIONS)[0]);model.load_state_dict(initial)
    require(state_hash(model)==q['initial_state_sha256'],'Actual initialization')
    config=read(p['config']);limits=robot_limits(config)
    require(native_domain_stats(y['base'],limits)['arm_outside_URDF_coordinates']==0,'Original base arm-domain exception')
    run=new_run('output_support',group=('effect_learning',));sources=dict(design['source_sha256'])
    files=[DESIGN,Path(__file__),ROOT/'cx_tta/output_support_adapter.py',ROOT/'tests/test_output_support.py',
        ROOT/'scripts/train_output_support.sh',Path(p['config'])]
    files += [RobotFK(config,side).urdf for side in ('left','right')]
    # Frozen reference proves gripper clamping occurs inside the native executor.
    robot_source=ROOT/'outputs/diagnostics/recovery_sources/turn_switch/source_20261002_153403_771011/robotwin/envs/robot/robot.py'
    require('gripper_val = np.clip(gripper_val, 0, 1)' in robot_source.read_text(),'Native gripper execution contract')
    files.append(robot_source);sources.update({str(f):digest(f) for f in files})
    snapshot=run/'source_snapshot';snapshot.mkdir()
    for f in files:
        if f.suffix in ('.py','.sh') and f.is_relative_to(ROOT/'cx_tta') or f.is_relative_to(ROOT/'scripts') or f.is_relative_to(ROOT/'tests'):
            (snapshot/f.name).write_bytes(f.read_bytes())
    protocol=dict(scope=SCOPE,status='frozen_before_training',design=str(DESIGN),parent_run=str(BASE),
        conditions=list(CONDITIONS),condition_spec={c:dict(output=m,clip_target=k) for c,(m,k) in CONDITIONS.items()},
        initial=str(BASE/'initial_seed0.pt'),initial_state_sha256=q['initial_state_sha256'],
        channel_dim=32,geometry_mean=q['geometry_mean'],parameters=q['parameters'],seed=0,updates=1400,
        lr=.0003,weight_decay=.01,auxiliary_coefficient=0.,rows=611,
        inputs=str(BASE/'inputs.npz'),labels=str(BASE/'labels.npz'),
        development_inputs=str(BASE/'development_inputs.npz'),development_labels=str(BASE/'development_labels.npz'),
        projection=p['projection'],robot_config=p['config'],reference_scales_native_units=q['caps_native_units'],
        row_order_and_weights='Original byte-identical611 rows, weights, masks and past; all rows every update',
        robot_joint_limits=limits.tolist(),time_limit_seconds=1200,inherited_source_gaps=33,
        robot_domain='No additional gripper clipping at policy boundary. Original per-tick native clipping/interpolation unchanged for all5 conditions. '
            'Reject nonfinite goals and arm targets outside same actual URDF bounds; record/refuse, never silently project/reset budget.',
        no_backbone_training=True,no_new_raw_data=True,no_router_or_TTA=True,
        source_sha256=sources)
    write_json(run/'training_protocol.json',protocol);print('RUN',run,flush=True);return run


def load(run):
    q=read(run/'training_protocol.json');require(q['scope']==SCOPE and q['conditions']==list(CONDITIONS),'Four-cell identity')
    require(q['updates']==1400 and q['rows']==611 and q['parameters']==42934,'Fixed budget/rows/allocation')
    verify(q['source_sha256']);p,old,x,y=original_load(BASE)
    require(q['initial_state_sha256']==old['initial_state_sha256'],'Shared actual init')
    return p,q,x,y


def model_from_initial(q,c,device='cpu'):
    model=OutputSupportAdapter(32,q['geometry_mean'],c).to(device)
    model.load_state_dict(torch.load(q['initial'],map_location=device,weights_only=True))
    require(state_hash(model)==q['initial_state_sha256'],'Init state bytes')
    return model


def loss(model,x,y):
    out=model(x,y['base'])
    return support_loss(out['delta'],y['base'],y['target'],y['valid'],y['row_weights'],model.caps,
        CONDITIONS[model.output_condition][1])


def preflight(run):
    p,q,xn,yn=load(run);x=tensors(xn);y=tensors(yn);checks=[]
    for c in CONDITIONS:
        model=model_from_initial(q,c)
        with torch.no_grad():out=model(x,y['base']);require(torch.equal(out['actions'],y['base']),'Initial zero identity')
        torch.manual_seed(901)
        with torch.no_grad():model.action_head[-1].weight.normal_(0,.01);model.action_head[-1].bias.normal_(0,.01)
        def gradients(labels=y,inputs=x):
            value=loss(model,inputs,labels)
            g=torch.autograd.grad(value,tuple(model.parameters()),allow_unused=True)
            return float(value.detach()),g
        before=gradients();masked={k:v.clone() for k,v in y.items()};masked['target'][~masked['valid']]=float('nan')
        after=gradients(masked)
        require(before[0]==after[0] and all((u is None and v is None) or torch.equal(u,v)
            for u,v in zip(before[1],after[1])),'Invalid tail gradient leaked')
        zero={k:v.clone() for k,v in y.items()};zero['row_weights'][0]=0.;baseline=gradients(zero)
        zero['target'][0]=float('nan');inputs={k:v.clone() for k,v in x.items()}
        for key in inputs:inputs[key][0]=0 if key=='has_history' else 17.
        changed=gradients(zero,inputs)
        require(baseline[0]==changed[0] and all((u is None and v is None) or torch.equal(u,v)
            for u,v in zip(baseline[1],changed[1])),'Zero row label/input leakage')
        require(all(g is None for (name,_),g in zip(model.named_parameters(),before[1])
            if name.startswith('geometry_head.')),'Unused geometry')
        old_error=None
        if c=='A_bounded_clipped':
            old=StructuredActionAdapter(32,q['geometry_mean']);old.load_state_dict(model.state_dict())
            old_loss=old_losses(old(x,y['base']),y,old.caps,0.)[0]
            old_g=torch.autograd.grad(old_loss,tuple(old.parameters()),allow_unused=True)
            require(float(old_loss)==before[0] and all((u is None and v is None) or torch.equal(u,v)
                for u,v in zip(before[1],old_g)),'Original loss/gradient not exact')
            old_error=0.
        checks.append(dict(condition=c,identity=True,initial_state_sha256=q['initial_state_sha256'],
            parameter_count=sum(v.numel() for v in model.parameters()),invalid_tail_exact=True,zero_row_exact=True,
            original_control_loss_gradient_error=old_error))
    write_json(run/'training_preflight.json',dict(passed=True,checks=checks,GPU_used=False,
        source_sha256={str(run/'training_protocol.json'):digest(run/'training_protocol.json')}))
    print('PREFLIGHT_PASS',flush=True)


def metadata(run,q,c):
    return dict(scope=SCOPE,condition=c,output_mode=CONDITIONS[c][0],clip_target=CONDITIONS[c][1],
        training_run=str(run),channels=32,geometry_mean=q['geometry_mean'],parameters=42934,seed=0,updates=1400,
        reference_scales_native_units=q['reference_scales_native_units'],projection_path=q['projection'],
        projection_sha256=digest(Path(q['projection'])),robot_config_path=q['robot_config'],
        robot_config_sha256=digest(Path(q['robot_config'])),initial_state_sha256=q['initial_state_sha256'],
        auxiliary_coefficient=0.,training_protocol_sha256=digest(run/'training_protocol.json'))


def compare_control(state):
    old=torch.load(BASE/'S0_seed0.pt',map_location='cpu',weights_only=True)['model']
    require(set(state)==set(old),'Control structure')
    return dict(exact=all(torch.equal(v.detach().cpu(),old[k]) for k,v in state.items()),
        max_abs=max(float((v.detach().cpu()-old[k]).abs().max()) for k,v in state.items()))


def train(run):
    p,q,xn,yn=load(run);require(read(run/'training_preflight.json')['passed'],'CPU gate')
    require(not (run/'training_environment.json').exists(),'Fresh training only')
    x=tensors(xn,'cuda');y=tensors(yn,'cuda');reports=[];start=time.monotonic()
    write_json(run/'training_environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),deterministic=torch.are_deterministic_algorithms_enabled(),
        parameters=42934,backbone_loaded=False,simulator_loaded=False))
    def timeout(*_):raise TimeoutError('Fixed four-model1200second compute bound')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(q['time_limit_seconds'])
    try:
        for c in CONDITIONS:
            model=model_from_initial(q,c,'cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            history=[]
            for step in range(1,1401):
                optimizer.zero_grad(set_to_none=True);value=loss(model,x,y)
                require(bool(torch.isfinite(value)),'Finite loss');value.backward();optimizer.step()
                if step==1 or step%200==0:
                    item=dict(step=step,loss=float(value.detach()));history.append(item)
                    replace_json(run/'training_progress.json',dict(condition=c,models_done=len(reports),**item));print(c,item,flush=True)
            model.eval()
            if c=='A_bounded_clipped':
                control=compare_control(model.state_dict());write_json(run/'control_reproduction.json',control)
                require(control['exact'],'Control must reproduce old S0 exactly before others')
            path=run/(c+'_seed0.pt');torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},metadata=metadata(run,q,c)),path)
            with torch.inference_mode():prediction={k:v.cpu().numpy() for k,v in model(x,y['base']).items() if k!='actions'}
            file=run/(c+'_predictions.npz');np.savez_compressed(file,**prediction)
            reports.append(dict(condition=c,updates=1400,initial_state_sha256=q['initial_state_sha256'],
                checkpoint=str(path),checkpoint_sha256=digest(path),predictions=str(file),predictions_sha256=digest(file),history=history))
        verify(q['source_sha256']);write_json(run/'training_result.json',dict(status='complete',reports=reports,
            elapsed_seconds=time.monotonic()-start,control_exact=True,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
        replace_json(run/'training_progress.json',dict(phase='complete',models_done=4));print('TRAIN_COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'training_failure.json',dict(completed_models=len(reports),traceback=traceback.format_exc()));raise
    finally:signal.alarm(0)


def read_checkpoint(path,condition,require_audit=True):
    path=Path(path).resolve();payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
    run=artifact_path(Path(meta['training_run']));_,q,_,_=load(run)
    require(condition in CONDITIONS and meta==metadata(run,q,condition),'Checkpoint identity')
    require(path==run/(condition+'_seed0.pt'),'Final checkpoint only')
    reports=read(run/'training_result.json');require(reports['status']=='complete' and len(reports['reports'])==4,'All4 completion')
    row=next(r for r in reports['reports'] if r['condition']==condition)
    require(digest(path)==row['checkpoint_sha256'],'Changed checkpoint')
    if require_audit:require(read(run/'training_audit.json')['passed'],'Independent CPU reload gate')
    return payload


def audit(run):
    p,q,xn,yn=load(run);x=tensors(xn);y=tensors(yn);xd=tensors(arrays(q['development_inputs']));yd=tensors(arrays(q['development_labels']))
    result=read(run/'training_result.json');require(result['status']=='complete','Training completion');reports=[]
    for row in result['reports']:
        c=row['condition'];payload=read_checkpoint(row['checkpoint'],c,False);model=OutputSupportAdapter(32,q['geometry_mean'],c).eval()
        model.load_state_dict(payload['model'])
        with torch.inference_mode():pred=model(x,y['base']);dev=model(xd,yd['base'])
        saved=arrays(row['predictions']);verify({row['predictions']:row['predictions_sha256']})
        maximum=max(float(np.abs(pred[k].numpy()-v).max()) for k,v in saved.items());require(maximum<2e-5,'CPU prediction reload')
        require(torch.equal(model(x,y['base'],False)['actions'],y['base']),'Exact bypass')
        if c.startswith(('A_','C_')):require(bool(torch.all(pred['delta'].abs()<=model.caps+1e-6)),'Declared bounded support')
        groups=[]
        for task,kind in sorted({(r['task'],r['kind']) for r in p['rows']}):
            ids=[i for i,r in enumerate(p['rows']) if (r['task'],r['kind'])==(task,kind)]
            w=y['row_weights'][ids];w=w/w.sum();v=y['valid'][ids];d=pred['delta'][ids]
            group=dict(task=task,kind=kind,rows=len(ids))
            for clipped,tag in ((True,'clipped'),(False,'raw')):
                group[tag+'_normalized_loss']=float(support_loss(d,y['base'][ids],y['target'][ids],v,w,model.caps,clipped))
            group.update(max_abs_arm_delta=float(d[...,list(range(6))+list(range(7,13))].abs().max()),
                **native_domain_stats(pred['actions'][ids].numpy(),q['robot_joint_limits']))
            groups.append(group)
        reports.append(dict(condition=c,cpu_reload_max_error=maximum,by_task_kind=groups,
            development_cup_delta_RMS=float(dev['delta'].square().mean().sqrt()),
            robot_domain=native_domain_stats(pred['actions'].numpy(),q['robot_joint_limits'])))
    control=compare_control(read_checkpoint(run/'A_bounded_clipped_seed0.pt','A_bounded_clipped',False)['model'])
    require(control['exact'],'Final control');verify(q['source_sha256'])
    write_json(run/'training_audit.json',dict(passed=True,reports=reports,control=control,
        GPU_used=False,offline_scores_are_not_success=True,source_sha256={str(run/'training_result.json'):digest(run/'training_result.json')}))
    print('AUDIT_PASS',json.dumps(reports,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('phase',choices=('prepare','preflight','train','audit'));ap.add_argument('--run',type=Path)
    a=ap.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if a.phase=='prepare':prepare();return
    if a.run is None:ap.error('--run required')
    {'preflight':preflight,'train':train,'audit':audit}[a.phase](output_path(a.run))


if __name__=='__main__':main()
