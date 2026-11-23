"""Independent isolation checks for S2e whole-scene paired training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paired_fixed_data import load
from .matched_recovery_data import load as old_load
from .fixed_recovery_dataset import verify_scene_disjoint
from .paths import write_json
from .audit import digest
from .probe_candidates import read,verify
from .time_encoding import TimeEncodedAdapter
from .train_rotation_curriculum import state_hash
from .train_scene_crossfit import candidate_loss
from .pose_adapter import TorchFK
from .robot_kinematics import RobotFK


def preflight(run):
    p,a,ctx,s=load(run);op,oa,oc,os=old_load(Path(p['source']));cp=read(Path(p['cache'])/'protocol.json')
    n=p['background_rows'];ids=s['train_indices'];rows=p['rows']
    assert len(rows)==758 and len(ids)==611 and n==735
    assert p['updates']==op['updates']==1400 and p['lr']==op['lr'] and p['initializations']==op['initializations']
    np.testing.assert_array_equal(ids[:588],os['train_indices'])
    np.testing.assert_array_equal(ctx[:n],oc)
    for k in oa:np.testing.assert_array_equal(a[k][:n],oa[k])
    for ci in range(len(p['conditions'])):
        np.testing.assert_array_equal(s['row_weights'][ci,:n],os['row_weights'][0])
        np.testing.assert_array_equal(s['pair_weights'][ci,:len(op['seam_pairs'])],os['pair_weights'][0])
    assert p['preprocess_source']==op['preprocess_source']
    # Reconstruct new coordinates directly; no refit, no recovery-fitted moments.
    with np.load(p['preprocess_source']) as z:pre={k:z[k].copy() for k in z.files}
    with np.load(Path(p['cache'])/'features.npz') as z:joint=z['joint'].astype(np.float32);state=z['state'].copy()
    state[:,[0,1,2,3,4,5,7,8,9,10,11,12]]/=np.pi
    visual=joint@pre['pca_components'].T-pre['pca_mean'][None]@pre['pca_components'].T
    current=np.concatenate([visual,state],1);history=np.array(cp['feature_history_indices']);past=np.zeros_like(current)
    for i,prev in enumerate(history):
        if prev>=0:
            assert prev<i
            cr,pr=cp['feature_rows'][i],cp['feature_rows'][prev]
            assert cr['scene_group']==pr['scene_group'] and cr['group']==pr['group']
            assert cr['native_count']-pr['native_count']==32
            past[i]=current[prev]
    rebuilt=(np.c_[current,past,(history>=0).astype(np.float32)]-pre['context_mean'])/pre['context_std']
    np.testing.assert_array_equal(ctx[n:],rebuilt[cp['feature_current_indices']])
    assert not ({rows[i]['group'] for i in ids}&{rows[i]['group'] for i in p['calibration_indices']})
    # Independently sum masses at each hierarchy level, including unequal fork counts.
    for collection,weights in ((rows[n:],s['row_weights'][0,n:]),
        (p['seam_pairs'][len(op['seam_pairs']):],s['pair_weights'][0,len(op['seam_pairs']):])):
        tasks=sorted({r['task'] for r in collection})
        for task in tasks:
            groups=sorted({r['group'] for r in collection if r['task']==task})
            for group in groups:
                branches=sorted({r['branch_group'] for r in collection if r['group']==group})
                for branch in branches:
                    ix=[i for i,r in enumerate(collection) if r['branch_group']==branch]
                    np.testing.assert_allclose(weights[ix],1/(len(tasks)*len(groups)*len(branches)*len(ix)),rtol=1e-6)
    local={g:i for i,g in enumerate(ids)}
    for e,pair in zip(p['seam_pairs'],s['pair_indices']):
        np.testing.assert_array_equal(pair,[local[e['before']],e['last'],local[e['after']]])
        before,after=rows[e['before']],rows[e['after']]
        assert before['group']==after['group']==e['group'] and a['valid'][e['before'],e['last']]
        if e['source_block']=='fixed_new':
            assert before['branch_group']==after['branch_group']==e['branch_group']
            assert before['frame']+32==after['frame'] and e['last']==31
    robots=[TorchFK(RobotFK(read(Path(p['config'])),side)).float() for side in ('left','right')]
    td={k:torch.from_numpy(v[ids]) for k,v in a.items()};context=torch.from_numpy(ctx[ids]);pairs=torch.from_numpy(s['pair_indices'])
    torch.manual_seed(21903);model=TimeEncodedAdapter(157,encoding='fourier',envelope=True)
    with torch.no_grad():model.head.weight.normal_(0,.01)
    assert not any(isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.Dropout)) for m in model.modules())
    checks=[]
    for ci,condition in enumerate(p['conditions']):
        omitted=p['omitted_groups'][condition];mask=np.array([rows[i]['group']==omitted for i in ids])
        wanted=s['row_weights'][0].copy();wanted[[i for i,r in enumerate(rows) if r['group']==omitted]]=0
        seam=s['pair_weights'][0].copy();seam[[i for i,e in enumerate(p['seam_pairs']) if e['group']==omitted]]=0
        np.testing.assert_array_equal(wanted,s['row_weights'][ci]);np.testing.assert_array_equal(seam,s['pair_weights'][ci])
        if omitted:
            verify_scene_disjoint([dict(rows[i],scene_seed=rows[i].get('scene_seed',int(rows[i]['group'].split(':')[-1])))
                for i in ids if wanted[i]>0], [rows[i] for i in ids if rows[i]['group']==omitted])
        weights=torch.from_numpy(wanted[ids]);pw=torch.from_numpy(seam)
        def calc(target,x=context,base=td['base']):
            loss=candidate_loss(model(x,base),target,td['valid'],weights,pairs,pw,robots,model.caps)
            gs=torch.autograd.grad(loss,tuple(model.parameters()))
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gs)
            return float(loss.detach()),[g.detach() for g in gs]
        value,grad=calc(td['target']);poison=td['target'].clone();poison[~td['valid']]=9.25;poison[mask]=9.25
        v,g=calc(poison);err=max(abs(value-v),max(float((x-y).abs().max()) for x,y in zip(grad,g)))
        assert err<=1e-6
        x=context.clone();x[mask]=19.25;b=td['base'].clone();b[mask]=1.25
        v,g=calc(poison,x,b);inputerr=max(abs(value-v),max(float((x-y).abs().max()) for x,y in zip(grad,g)))
        assert inputerr<=1e-6
        changed=td['target'].clone();active=next(j for j,i in enumerate(ids) if wanted[i]>0 and rows[i]['source_block']=='fixed_new')
        changed[active,0,0]+=.1;_,g=calc(changed)
        sensitivity=max(float((x-y).abs().max()) for x,y in zip(grad,g));assert sensitivity>1e-6
        checks.append(dict(condition=condition,held_rows=int(mask.sum()),loss_gradient_error=err,
            held_input_gradient_error=inputerr,positive_control_gradient_change=sensitivity,
            background_mass=float(wanted[:n].sum()),new_mass=float(wanted[n:].sum()),seam_mass=float(seam.sum())))
    for initial in p['initializations']:
        model=TimeEncodedAdapter(157,encoding=initial['encoding'],envelope=True)
        model.load_state_dict(torch.load(initial['path'],map_location='cpu',weights_only=True))
        assert state_hash(model)==initial['state_sha256'] and sum(v.numel() for v in model.parameters())==41742
    verify(p)
    write_json(run/'preflight_audit.json',dict(passed=True,checks=checks,train_rows=len(ids),calibration_rows=len(p['calibration_indices']),
        coordinate_reprojection_exact=True,background_unchanged=True,original_pca_unchanged=True,
        source_sha256={str(run/'protocol.json'):digest(run/'protocol.json'),str(Path(__file__)):digest(Path(__file__))}))
    print('PREFLIGHT_PASSED',checks,flush=True)


def predictions(run):
    p,a,ctx,s=load(run);result=read(run/'result.json');assert result['status']=='complete'
    assert read(run/'preflight_audit.json')['passed'];seen=set();maximum=0.
    for report in result['reports']:
        key=tuple(report[k] for k in ('condition','encoding','seed'));assert key not in seen;seen.add(key)
        assert report['updates']==p['updates']==1400
        initial=next(x for x in p['initializations'] if (x['encoding'],x['seed'])==key[1:])
        assert report['initial_sha256']==initial['state_sha256']
        for field,h in (('checkpoint','checkpoint_sha256'),('artifact','sha256')):assert digest(Path(report[field]))==report[h]
        with np.load(report['artifact']) as z:pred=z['predicted'].copy();np.testing.assert_array_equal(z['indices'],np.arange(len(ctx)))
        assert np.isfinite(pred).all()
        model=TimeEncodedAdapter(157,encoding=report['encoding'],envelope=True)
        model.load_state_dict(torch.load(report['checkpoint'],map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():again=np.concatenate([model(torch.from_numpy(ctx[i:i+64]),torch.from_numpy(a['base'][i:i+64])).numpy() for i in range(0,len(ctx),64)])
        error=float(np.max(np.abs(again-pred)));maximum=max(maximum,error);assert error<=2e-5
        lower=np.minimum(a['base'][...,[6,13]],0);upper=np.maximum(a['base'][...,[6,13]],1)
        assert np.all(pred[...,[6,13]]>=lower-1e-6) and np.all(pred[...,[6,13]]<=upper+1e-6)
    assert seen=={(c,e,i) for c in p['conditions'] for e in p['encodings'] for i in p['seeds']}
    verify(p)
    write_json(run/'candidate_audit.json',dict(passed=True,models=len(seen),cpu_prediction_max_error=maximum,
        source_sha256={str(run/'result.json'):digest(run/'result.json'),str(Path(__file__)):digest(Path(__file__))}))
    print('CANDIDATE_AUDIT_PASSED',len(seen),maximum,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);threadpool_limits(2)
    (preflight if args.preflight else predictions)(args.run.resolve())
