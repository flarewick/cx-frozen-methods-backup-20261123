"""Actual-data isolation, fixed weights/initialization and CPU checkpoint checks."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read,verify
from .matched_recovery_data import load
from .time_encoding import TimeEncodedAdapter
from .train_rotation_curriculum import state_hash
from .train_scene_crossfit import candidate_loss
from .pose_adapter import TorchFK
from .robot_kinematics import RobotFK


def balanced(rows):
    tasks={r['task'] for r in rows};counts=Counter(r['group'] for r in rows)
    return np.array([1/(len(tasks)*len({q['group'] for q in rows if q['task']==r['task']})*counts[r['group']]) for r in rows],np.float32)


def preflight(run):
    p,a,ctx,s=load(run);parent=read(Path(p['source'])/'protocol.json')
    ids=s['train_indices'];rows=p['rows'];expected=np.zeros(len(rows),np.float32)
    assert len(ids)==588 and len(rows)==735
    for kind in ('expert','anchor','recovery'):
        ki=[i for i in ids if rows[i]['kind']==kind];expected[ki]=balanced([rows[i] for i in ki])
    np.testing.assert_array_equal(expected,s['row_weights'][0])
    original_pairs=balanced(p['seam_pairs']);np.testing.assert_array_equal(original_pairs,s['pair_weights'][0])
    with np.load(p['preprocess_source']) as z:
        np.testing.assert_array_equal(z['fit_indices'],p['preprocess_expert_source_indices'])
        assert all(parent['rows'][i]['kind']=='expert' for i in z['fit_indices'])
        for i in p['source_indices']:
            prev=int(z['history_indices'][i])
            if prev>=0:
                assert prev in p['source_indices'] and parent['rows'][i]['group']==parent['rows'][prev]['group']
                assert parent['rows'][prev]['frame']<parent['rows'][i]['frame']
    assert not set(p['source_indices'])&set(parent['parts']['validation'])
    assert not ({rows[i]['group'] for i in ids}&{rows[i]['group'] for i in p['calibration_indices']})
    checks=[];robots=[TorchFK(RobotFK(read(Path(p['config'])),side)).float() for side in ('left','right')]
    td={k:torch.from_numpy(v[ids]) for k,v in a.items()};context=torch.from_numpy(ctx[ids]);pairs=torch.from_numpy(s['pair_indices'])
    torch.manual_seed(21903);model=TimeEncodedAdapter(157,encoding='fourier',envelope=True)
    with torch.no_grad():model.head.weight.normal_(0,.01)
    assert not any(isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.Dropout)) for m in model.modules())
    for ci,condition in enumerate(p['conditions']):
        omitted=p['omitted_groups'][condition];mask=np.array([rows[i]['group']==omitted for i in ids])
        wanted=expected.copy();wanted[[i for i,r in enumerate(rows) if r['group']==omitted]]=0
        np.testing.assert_array_equal(wanted,s['row_weights'][ci])
        seam=original_pairs.copy();seam[[j for j,e in enumerate(p['seam_pairs']) if e['group']==omitted]]=0
        np.testing.assert_array_equal(seam,s['pair_weights'][ci])
        weights=torch.from_numpy(s['row_weights'][ci,ids]);pw=torch.from_numpy(seam)
        def calculation(target,x=context):
            prediction=model(x,td['base'])
            loss=candidate_loss(prediction,target,td['valid'],weights,pairs,pw,robots,model.caps)
            gradients=torch.autograd.grad(loss,tuple(model.parameters()))
            return float(loss.detach()),[g.detach() for g in gradients]
        base,gradient=calculation(td['target'])
        poisoned=td['target'].clone();poisoned[~td['valid']]=9.25
        if omitted is not None:poisoned[mask]=9.25
        again,other=calculation(poisoned)
        gd=max(float((u-v).abs().max()) for u,v in zip(gradient,other))
        if abs(base-again)>1e-6 or gd>1e-6:raise ValueError('Omitted/tail targets affected loss or gradients')
        input_error=0.
        if omitted is not None:
            x=context.clone();x[mask]=19.25
            value,gg=calculation(poisoned,x)
            input_error=max(abs(base-value),max(float((u-v).abs().max()) for u,v in zip(gradient,gg)))
            if input_error>1e-6:raise ValueError('Zero-weight rows coupled to active training')
        # A real active teacher-label perturbation must change the gradient (positive sensitivity control).
        changed=td['target'].clone();active=next(i for i,v in enumerate(weights) if v>0 and rows[ids[i]]['kind']=='recovery')
        changed[active,0,0]+=.1;_,control=calculation(changed)
        sensitivity=max(float((u-v).abs().max()) for u,v in zip(gradient,control))
        if not sensitivity>1e-6:raise ValueError('Loss ignores active recovery labels')
        checks.append(dict(condition=condition,excluded_rows=int(mask.sum()),tail_and_excluded_loss_error=abs(base-again),
            tail_and_excluded_gradient_error=gd,excluded_input_gradient_error=input_error,
            active_target_gradient_change=sensitivity,
            masses={kind:float(wanted[[i for i,r in enumerate(rows) if r['kind']==kind]].sum()) for kind in ('expert','anchor','recovery')},
            seam_mass=float(seam.sum())))
    old=read(Path(p['source'])/'result.json')['reports'];initial=[]
    for record in p['initializations']:
        m=TimeEncodedAdapter(157,encoding=record['encoding'],envelope=True)
        m.load_state_dict(torch.load(record['path'],map_location='cpu',weights_only=True))
        previous=next(r for r in old if (r['job'],r['encoding'],r['seed'])==('outer',record['encoding'],record['seed']))
        assert state_hash(m)==record['state_sha256']==previous['initial_sha256']
        assert sum(v.numel() for v in m.parameters())==p['parameters']
        initial.append(dict(encoding=record['encoding'],seed=record['seed'],state_sha256=state_hash(m)))
    verify(p)
    write_json(run/'preflight_audit.json',dict(passed=True,checks=checks,initializations=initial,
        train_rows=588,calibration_rows=147,old_validation_rows=0,preprocessing_unchanged=True,
        source_sha256={str(run/'protocol.json'):digest(run/'protocol.json'),str(Path(__file__)):digest(Path(__file__))}))
    print('PREFLIGHT_PASSED',checks,flush=True)


def predictions(run):
    p,a,ctx,s=load(run);result=read(run/'result.json')
    assert read(run/'preflight_audit.json')['passed']
    expected={(c,e,i) for c in p['conditions'] for e in p['encodings'] for i in p['seeds']};seen=set();maximum=0.;replays=[]
    old=read(Path(p['source'])/'result.json')['reports']
    for report in result['reports']:
        key=tuple(report[k] for k in ('condition','encoding','seed'));assert key not in seen;seen.add(key)
        assert report['updates']==1400
        initial=next(x for x in p['initializations'] if (x['encoding'],x['seed'])==key[1:])
        assert report['initial_sha256']==initial['state_sha256']
        for field,h in (('checkpoint','checkpoint_sha256'),('artifact','sha256')):assert digest(Path(report[field]))==report[h]
        with np.load(report['artifact']) as z:pred=z['predicted'].copy();np.testing.assert_array_equal(z['indices'],np.arange(len(ctx)))
        model=TimeEncodedAdapter(157,encoding=report['encoding'],envelope=True)
        state=torch.load(report['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(state);model.eval()
        with torch.no_grad():reloaded=np.concatenate([model(torch.from_numpy(ctx[i:i+64]),torch.from_numpy(a['base'][i:i+64])).numpy() for i in range(0,len(ctx),64)])
        error=float(np.max(np.abs(reloaded-pred)));maximum=max(maximum,error)
        if error>2e-5:raise ValueError('GPU saved prediction differs from CPU reload')
        lower=np.minimum(a['base'][...,[6,13]],0);upper=np.maximum(a['base'][...,[6,13]],1)
        if np.any(pred[...,[6,13]]<lower-1e-6) or np.any(pred[...,[6,13]]>upper+1e-6):raise ValueError('Gripper envelope broken')
        if report['condition']=='both':
            original=next(x for x in old if (x['job'],x['encoding'],x['seed'])==('outer',report['encoding'],report['seed']))
            previous=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
            difference=max(float((state[k]-previous[k]).abs().max()) for k in state)
            replays.append(dict(encoding=report['encoding'],seed=report['seed'],maximum_weight_difference=difference,
                                bitwise_same=all(torch.equal(state[k],previous[k]) for k in state)))
    assert seen==expected
    verify(p)
    write_json(run/'candidate_audit.json',dict(passed=True,models=len(seen),cpu_prediction_max_error=maximum,
        original_outer_replays=replays,source_sha256={str(run/'result.json'):digest(run/'result.json'),str(Path(__file__)):digest(Path(__file__))},
        scope='Isolation and numerical audit. The both-condition replay comparison is reported, not silently assumed identical.'))
    print('CANDIDATE_AUDIT_PASSED',maximum,replays,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);threadpool_limits(2)
    (preflight if args.preflight else predictions)(args.run.resolve())
