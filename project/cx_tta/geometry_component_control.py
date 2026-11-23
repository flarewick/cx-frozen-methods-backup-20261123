"""S2i: frozen-data factorial expert-position/rotation gradient diagnostic."""
import argparse
import copy
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import new_run, write_json, replace_json
from .audit import digest
from .probe_candidates import read, verify
from .geometry_readout import GeometryReadout, geometry_loss, load, metrics
from .train_rotation_curriculum import state_hash


COEFFICIENTS = {
    'recovery_only_omit': (0., 0.),
    'expert_position_omit': (1., 0.),
    'expert_rotation_omit': (0., 1.),
    'expert_plus_omit': (1., 1.),
}
NEW_CONDITIONS = ('expert_position_omit', 'expert_rotation_omit')


def component_loss(pred, target, valid, weights, expert_rows, position, rotation):
    """Source-side supervision only; keep ORIGINAL valid-dimension denominator."""
    selector = torch.ones_like(pred)
    selector[expert_rows, :12] = position
    selector[expert_rows, 12:] = rotation
    per_row = ((pred-target).square()*valid*selector).sum(1)/valid.sum(1).clamp_min(1)
    return (per_row*weights).sum()


def prepare(parent):
    parent = parent.resolve()
    p, q, _, _ = load(parent)
    for name in ('completion.json', 'learning_preflight.json', 'learning_audit.json'):
        d = read(parent/name); verify(d)
        assert d.get('passed', d.get('status') == 'complete_audited')
    result = read(parent/'learning_result.json')
    assert result['status'] == 'complete' and q['updates'] == 1400 and len(result['reports']) == 27
    for r in result['reports']:
        assert digest(Path(r['checkpoint'])) == r['checkpoint_sha256']
        assert digest(Path(r['artifact'])) == r['sha256']
    run = new_run('geometry_components', group=('feature_caches',))
    for name in ('inputs.npz', 'labels_offline_only.npz'):
        (run/name).symlink_to((parent/name).resolve())
    pp = copy.deepcopy(p); pp['component_control_parent'] = str(parent)
    pp['source_sha256'].update({str(parent/'protocol.json'):digest(parent/'protocol.json')})
    write_json(run/'protocol.json', pp)
    jobs = []
    for j in q['jobs']:
        if j['condition'] != 'expert_plus_omit': continue
        for condition in NEW_CONDITIONS:
            job = copy.deepcopy(j); job['condition'] = condition
            job['expert_position_coefficient'], job['expert_rotation_coefficient'] = COEFFICIENTS[condition]
            jobs.append(job)
    qq = copy.deepcopy(q); qq['jobs'] = jobs; qq['conditions'] = list(NEW_CONDITIONS)
    qq['reference_conditions'] = ['recovery_only_omit', 'expert_plus_omit']
    qq['component_control_parent'] = str(parent)
    qq['loss'] = 'Original full-valid per-row denominator and weights; source-side expert xyz columns0:12 and root6D columns12:24 multiplied by fixed0/1. ALL recovery target columns and coefficients unchanged.'
    qq['evaluation'] = 'Final1400,18new heads plus18immutable parent references; all3scenes and3seeds. No action/task-success interpretation.'
    qq['criterion'] = 'Retain parent joint-vs-recovery development criterion. New primary causal diagnostics: face(J)-face(P),face(O)-face(R),face(P)-face(R),face(J)-face(O); >1degree is harm, all3seeds in >=2scenes is consistent. Rotation-specific harm requires BOTH rotation contrasts consistent in the same >=2scenes; otherwise report mixed interactions. Also report full SO3, root/function/relative position and training fit. No seed/head/budget selection.'
    files = [parent/f for f in ('learning_protocol.json','learning_result.json','summary.json','completion.json')]
    files += [run/f for f in ('protocol.json','inputs.npz','labels_offline_only.npz')]
    files += [Path(__file__), Path(__file__).parents[1]/'scripts/train_geometry_components.sh',
              Path(__file__).parents[1]/'tests/test_geometry_component_control.py']
    qq['source_sha256'].update({str(f):digest(f) for f in files})
    write_json(run/'learning_protocol.json', qq)
    write_json(run/'component_protocol.json', dict(status='frozen_before_training', parent=str(parent),
        new_models=18, reused_models=18, updates=1400, seeds=q['seeds'], coefficients=COEFFICIENTS,
        unchanged='Same arrays/PCA/history flags, 9actual initializations, shared valid-label prior, architecture, AdamW hyperparameters, row weights and full-valid denominator; all recovery supervision unchanged. Only source-side expert position/rotation factors differ.',
        primary=qq['criterion'], time_limit_seconds=600,
        limitation='Only3reused switch developer scenes. Shared prior still contains expert pose/position means even when its gradient coefficient is0. No new action model, simulator, evaluation, router or TTA.',
        source_sha256={str(run/'learning_protocol.json'):digest(run/'learning_protocol.json')}))
    (run/'source_snapshot').mkdir()
    for f in files[-3:]: (run/'source_snapshot'/f.name).write_bytes(f.read_bytes())
    print('RUN', run, flush=True)


def preflight(run):
    p,q,raw,labels = load(run); parent=Path(q['component_control_parent']); pq=read(parent/'learning_protocol.json')
    x={k:torch.from_numpy(v) for k,v in raw.items()}; y=torch.from_numpy(labels['regression']); valid=torch.from_numpy(labels['regression_valid'])
    er=torch.zeros(len(y),dtype=torch.bool); er[p['expert_indices']]=True
    assert q['initializations']==pq['initializations'] and q['updates']==pq['updates']==1400
    for k in ('lr','weight_decay','parameters','expert_weights','recovery_weights','common_initial_prior'):
        assert q[k]==pq[k],k
    checks=[]
    def compare(a,b):
        assert all((v is None)==(w is None) for v,w in zip(a[1],b[1]))
        return max([abs(a[0]-b[0])]+[float((v-w).abs().max()) for v,w in zip(a[1],b[1]) if v is not None])
    for job in q['jobs']:
        original=next(j for j in pq['jobs'] if j['scene']==job['scene'] and j['condition']=='expert_plus_omit')
        for k in ('weights','held','held_weights','mean','train'):assert job[k]==original[k],k
        assert set(job['train']).isdisjoint(job['held'])
        assert not any(p['rows'][i]['scene_group']==job['scene'] for i in job['train'])
        for seed in q['seeds']:
            init=next(i for i in q['initializations'] if (i['scene'],i['seed'])==(job['scene'],seed))
            assert digest(Path(init['path']))==init['sha256']
            model=GeometryReadout(p['channel_dim'],job['mean'])
            model.load_state_dict(torch.load(init['path'],map_location='cpu',weights_only=True))
            assert state_hash(model)==init['state_sha256']
            with torch.no_grad():np.testing.assert_allclose(model(x)['regression'],np.tile(job['mean'],(len(y),1)),atol=5e-7,rtol=0)
        torch.manual_seed(8301); model=GeometryReadout(p['channel_dim'],job['mean'])
        with torch.no_grad():model.head.weight.normal_(0,.03);model.query.weight.normal_(0,.01)
        assert not any(isinstance(v,(torch.nn.BatchNorm1d,torch.nn.Dropout)) for v in model.modules())
        weights=torch.tensor(job['weights'],dtype=torch.float32); pos,rot=COEFFICIENTS[job['condition']]
        def calc(data=x,target=y,mask=valid,factors=(pos,rot),old_weights=None):
            pred=model(data)['regression']
            loss=(component_loss(pred,target,mask,weights,er,*factors) if old_weights is None else geometry_loss(pred,target,mask,old_weights))
            grads=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
            return float(loss.detach()),[None if v is None else v.detach() for v in grads]
        base=calc(); zero=np.flatnonzero(np.array(job['weights'])==0)
        xx={k:v.clone() for k,v in x.items()}
        for v in xx.values():v[zero]=19.25
        yy=y.clone();yy[zero]=123.;vv=valid.clone();vv[zero]=True
        isolated=compare(base,calc(xx,yy,vv));assert isolated<1e-6
        yy=y.clone();yy[~valid]=999.;masked=compare(base,calc(target=yy));assert masked<1e-6
        yy=y.clone(); inactive=slice(12,24) if pos else slice(0,12);yy[er,inactive]+=999.
        removed=compare(base,calc(target=yy));assert removed<1e-6
        yy=y.clone();col=0 if pos else 12;yy[p['expert_indices'][0],col]+=.1
        active=compare(base,calc(target=yy));assert active>1e-6
        yy=y.clone();rec=next(i for i in p['recovery_indices'] if job['weights'][i]>0);yy[rec,0]+=.1;yy[rec,12]+=.1
        recovery=compare(base,calc(target=yy));assert recovery>1e-6
        joint=compare(calc(factors=(1.,1.)),calc(old_weights=weights));assert joint<1e-6
        control=next(j for j in pq['jobs'] if j['scene']==job['scene'] and j['condition']=='recovery_only_omit')
        none=compare(calc(factors=(0.,0.)),calc(old_weights=torch.tensor(control['weights'],dtype=torch.float32)));assert none<1e-6
        checks.append(dict(scene=job['scene'],condition=job['condition'],held_input_target_gradient_error=isolated,
            invalid_target_gradient_error=masked,removed_term_gradient_error=removed,active_expert_gradient_change=active,
            recovery_gradient_change=recovery,joint_parent_loss_gradient_error=joint,recovery_parent_loss_gradient_error=none))
    verify(p);verify(q)
    write_json(run/'learning_preflight.json',dict(passed=True,checks=checks,new_models=18,paired_initializations=True,
        full_valid_denominator_unchanged=True,parent_joint_and_recovery_gradient_parity=True,
        source_sha256={str(run/'learning_protocol.json'):digest(run/'learning_protocol.json'),str(Path(__file__)):digest(Path(__file__))}))
    print('COMPONENT_PREFLIGHT_PASSED',len(checks),flush=True)


def train(run,device):
    p,q,raw,labels=load(run);assert read(run/'learning_preflight.json')['passed']
    if (run/'learning_environment.json').exists():raise ValueError('Fresh run required')
    x={k:torch.from_numpy(v).to(device) for k,v in raw.items()};y=torch.from_numpy(labels['regression']).to(device);valid=torch.from_numpy(labels['regression_valid']).to(device)
    er=torch.zeros(len(y),dtype=torch.bool,device=device);er[p['expert_indices']]=True
    write_json(run/'learning_environment.json',dict(torch=torch.__version__,device=device,threads=torch.get_num_threads(),
        gpu=None if device=='cpu' else torch.cuda.get_device_name(0),deterministic=torch.are_deterministic_algorithms_enabled(),
        parameters=q['parameters'],backbone_loaded=False,simulator_loaded=False))
    start=time.monotonic();reports=[]
    def timeout(*_):raise TimeoutError('Component control exceeded600seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        for job in q['jobs']:
            weights=torch.tensor(job['weights'],dtype=torch.float32,device=device);pos,rot=COEFFICIENTS[job['condition']]
            for seed in q['seeds']:
                init=next(v for v in q['initializations'] if (v['scene'],v['seed'])==(job['scene'],seed))
                model=GeometryReadout(p['channel_dim'],job['mean']).to(device)
                model.load_state_dict(torch.load(init['path'],map_location=device,weights_only=True));assert state_hash(model)==init['state_sha256']
                optimizer=torch.optim.AdamW(model.parameters(),lr=q['lr'],weight_decay=q['weight_decay'])
                history=[];tag=job['scene'].split(':')[-1]+'_'+job['condition']+'_s'+str(seed)
                for step in range(1,q['updates']+1):
                    optimizer.zero_grad();loss=component_loss(model(x)['regression'],y,valid,weights,er,pos,rot)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();optimizer.step()
                    if step==1 or step%200==0:
                        history.append(dict(step=step,loss=float(loss.detach())))
                        replace_json(run/'learning_progress.json',dict(tag=tag,update=step,completed_models=len(reports),elapsed_seconds=time.monotonic()-start))
                model.eval();checkpoint=run/(tag+'.pt');torch.save(model.state_dict(),checkpoint)
                with torch.no_grad():outputs={k:v.cpu().numpy() for k,v in model(x).items()}
                artifact=run/(tag+'.npz');np.savez_compressed(artifact,**outputs)
                reports.append(dict(scene=job['scene'],condition=job['condition'],seed=seed,initial_sha256=init['state_sha256'],updates=q['updates'],
                    history=history,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),artifact=str(artifact),sha256=digest(artifact)))
                print('COMPONENT_MODEL',tag,len(reports),'loss',float(loss.detach()),flush=True)
        verify(p);verify(q)
        write_json(run/'learning_result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-start,
            peak_gpu_allocated_bytes=None if device=='cpu' else torch.cuda.max_memory_allocated(),
            scope='18small readout heads; original1400joint/control references remain immutable. No action/task-success result.'))
        replace_json(run/'learning_progress.json',dict(phase='complete',models=len(reports)))
        print('COMPONENT_TRAINING_COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'learning_failure.json',dict(completed_models=len(reports),traceback=traceback.format_exc()));raise
    finally:signal.alarm(0)


def finalize(run):
    p,q,raw,labels=load(run);parent=Path(q['component_control_parent']);pq=read(parent/'learning_protocol.json')
    result=read(run/'learning_result.json');assert result['status']=='complete' and len(result['reports'])==18
    x={k:torch.from_numpy(v) for k,v in raw.items()};reports=[];maximum=0.
    for record in result['reports']:
        job=next(j for j in q['jobs'] if (j['scene'],j['condition'])==(record['scene'],record['condition']))
        init=next(i for i in q['initializations'] if (i['scene'],i['seed'])==(record['scene'],record['seed']))
        assert record['initial_sha256']==init['state_sha256'] and record['updates']==1400
        assert digest(Path(record['checkpoint']))==record['checkpoint_sha256'] and digest(Path(record['artifact']))==record['sha256']
        model=GeometryReadout(p['channel_dim'],job['mean']);model.load_state_dict(torch.load(record['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():again={k:v.numpy() for k,v in model(x).items()}
        with np.load(record['artifact']) as z:stored={k:z[k].copy() for k in z.files}
        assert set(stored)==set(again) and all(np.isfinite(v).all() for v in stored.values())
        error=max(float(np.abs(again[k]-stored[k]).max()) for k in again);assert error<2e-5;maximum=max(maximum,error)
        for k in ('current_attention','past_attention'):
            assert (stored[k]>=0).all();np.testing.assert_allclose(stored[k].sum(-1),4,atol=2e-6)
        r={k:record[k] for k in ('scene','condition','seed')}
        r['held']=metrics(stored['regression'],labels,job['held'],job['held_weights'])
        r['train']=metrics(stored['regression'],labels,job['train'],np.array(job['weights'])[job['train']])
        r['expert_by_task']={}
        for task in sorted({p['rows'][i]['task'] for i in p['expert_indices']}):
            ids=[i for i in p['expert_indices'] if p['rows'][i]['task']==task]
            r['expert_by_task'][task]=metrics(stored['regression'],labels,ids,np.array(q['expert_weights'])[ids])
        ids=job['held'];ee=labels['command_ee_world'][ids,1];point=stored['regression'][ids,6:9]*.1
        relative=np.einsum('ni,nij->nj',point-ee[:,:3,3],ee[:,:3,:3])
        cm=float(np.array(job['held_weights'])@np.linalg.norm(relative-labels['functional_relative_command_m'][ids,1,0],axis=1)*100)
        assert abs(cm-r['held']['primary']['functional_position_cm'])<2e-5
        r['held_right_command_relative_functional_cm']=cm;r['source']='new_final1400';reports.append(r)
    parent_summary=read(parent/'summary.json');parent_result=read(parent/'learning_result.json')
    for r in parent_summary['reports']:
        if r['condition'] not in q['reference_conditions']:continue
        rr=copy.deepcopy(r);rr['source']='immutable_parent_final1400';reports.append(rr)
        record=next(v for v in parent_result['reports'] if (v['scene'],v['condition'],v['seed'])==(r['scene'],r['condition'],r['seed']))
        assert digest(Path(record['checkpoint']))==record['checkpoint_sha256'] and digest(Path(record['artifact']))==record['sha256']
        job=next(j for j in pq['jobs'] if (j['scene'],j['condition'])==(r['scene'],r['condition']))
        with np.load(record['artifact']) as z:pred=z['regression']
        assert metrics(pred,labels,job['held'],job['held_weights'])==r['held']
    expected={(j['scene'],c,s) for j in q['jobs'] for c in COEFFICIENTS for s in q['seeds']}
    assert len(reports)==36 and {(r['scene'],r['condition'],r['seed']) for r in reports}==expected
    contrasts=[];signals=[]
    for scene in sorted({j['scene'] for j in q['jobs']}):
        baseline=next(b for b in parent_summary['baselines'] if b['scene']==scene)['held']['primary']
        for seed in q['seeds']:
            values={c:next(r for r in reports if (r['scene'],r['condition'],r['seed'])==(scene,c,seed))['held']['primary'] for c in COEFFICIENTS}
            for name,a,b in [('rotation_given_position','expert_plus_omit','expert_position_omit'),('rotation_without_position','expert_rotation_omit','recovery_only_omit'),
                             ('position_without_rotation','expert_position_omit','recovery_only_omit'),('position_given_rotation','expert_plus_omit','expert_rotation_omit')]:
                contrast={k:values[a][k]-values[b][k] for k in ('root_position_cm','functional_position_cm','root_angle_deg','face_angle_deg')}
                contrasts.append(dict(scene=scene,seed=seed,contrast=name,delta=contrast,face_harm_gt1=contrast['face_angle_deg']>1))
        for c in COEFFICIENTS:
            passing=[]
            for seed in q['seeds']:
                a=next(r for r in reports if (r['scene'],r['condition'],r['seed'])==(scene,c,seed))['held']['primary']
                b=next(r for r in reports if (r['scene'],r['condition'],r['seed'])==(scene,'recovery_only_omit',seed))['held']['primary']
                passing.append(all(a[k]<b[k] and a[k]<baseline[k] for k in ('root_position_cm','functional_position_cm')) and a['face_angle_deg']<=b['face_angle_deg']+1)
            signals.append(dict(scene=scene,condition=c,paired_seed_pass=passing,consistent_improvement=all(passing)))
    rotation_harm=[]
    for scene in sorted({j['scene'] for j in q['jobs']}):
        c=[r for r in contrasts if r['scene']==scene and r['contrast'].startswith('rotation_')]
        rotation_harm.append(dict(scene=scene,both_rotation_contrasts_harm_all3seeds=all(r['face_harm_gt1'] for r in c)))
    verify(p);verify(q);verify(read(parent/'completion.json'))
    sources={str(run/f):digest(run/f) for f in ('learning_protocol.json','learning_result.json','learning_preflight.json')}
    sources.update({str(parent/f):digest(parent/f) for f in ('learning_protocol.json','learning_result.json','summary.json','completion.json')})
    sources[str(Path(__file__))]=digest(Path(__file__))
    write_json(run/'learning_audit.json',dict(passed=True,new_models=18,reference_models=18,cpu_prediction_max_error=maximum,
        relative_position_metrics_parity=True,parent_recomputed_held_metrics_parity=True,source_sha256=sources))
    write_json(run/'summary.json',dict(status='complete',reports=reports,contrasts=contrasts,scene_signals=signals,
        baselines=parent_summary['baselines'],rotation_specific_harm=rotation_harm,
        consistent_rotation_specific_harm=sum(v['both_rotation_contrasts_harm_all3seeds'] for v in rotation_harm)>=2,
        development_signal_by_condition={c:sum(v['consistent_improvement'] for v in signals if v['condition']==c)>=2 for c in COEFFICIENTS},
        scope='Fixed1400 factorial expert-gradient diagnostic,3reused developer scenes, all seeds. No task success or action-adapter result.',
        source_sha256={str(run/'learning_audit.json'):digest(run/'learning_audit.json')}))
    lines=['# S2i fixed1400 expert-supervision components','','All3seeds reported; immutable references reused. Expert metrics use training-side observations; removed target components are not supervised.','',
           '|Scene|Condition|Root cm (min–max)|Functional cm (min–max)|Root SO3°|Face°|Gates|','|---|---|---|---|---|---|---|']
    for scene in sorted({j['scene'] for j in q['jobs']}):
        for c in COEFFICIENTS:
            vals=[r['held']['primary'] for r in reports if (r['scene'],r['condition'])==(scene,c)]
            root=[v['root_position_cm'] for v in vals];fun=[v['functional_position_cm'] for v in vals]
            lines.append(f'|{scene}|{c}|{np.mean(root):.2f} ({min(root):.2f}–{max(root):.2f})|{np.mean(fun):.2f} ({min(fun):.2f}–{max(fun):.2f})|{np.mean([v["root_angle_deg"] for v in vals]):.2f}|{np.mean([v["face_angle_deg"] for v in vals]):.2f}|{sum(v["geometry_gate"] for v in vals)}/3|')
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
    print('COMPONENT_FINAL_AUDIT_PASSED',maximum,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('path',type=Path)
    parser.add_argument('--phase',choices=('prepare','preflight','train','final'),required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if a.phase=='train':train(a.path.resolve(),a.device)
    else:{'prepare':prepare,'preflight':preflight,'final':finalize}[a.phase](a.path.resolve())
