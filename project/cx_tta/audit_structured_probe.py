"""Isolation, input provenance and final CPU replay for the S2g diagnostic head."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json
from .probe_candidates import read,verify
from .audit import digest
from .structured_readout_probe import StructuredProbe,load_probes,probe_loss,metrics
from .train_rotation_curriculum import state_hash


def preflight(run):
    p,q,inputs,labels=load_probes(run)
    allowed={'visual','text','state','past_visual','past_text','past_state','has_history','ee_pose','past_ee_pose'}
    assert set(inputs)==allowed and all(np.isfinite(x).all() for x in inputs.values())
    assert len(q['rows'])==20 and q['updates']==600 and len(q['jobs'])==3
    x={k:torch.from_numpy(v) for k,v in inputs.items()};reg=torch.from_numpy(labels['regression']);phase=torch.from_numpy(labels['phase'])
    checks=[]
    for job in q['jobs']:
        train=job['train'];held=job['held'];tw=np.array(job['train_weights']);weights=torch.tensor(tw,dtype=torch.float32)
        assert set(train).isdisjoint(held) and set(train+held)==set(range(20))
        assert not ({q['rows'][i]['scene_group'] for i in train}&{q['rows'][i]['scene_group'] for i in held})
        np.testing.assert_allclose(np.array(job['mean']),tw@labels['regression'][train],atol=0,rtol=0)
        for record in [i for i in q['initializations'] if i['scene']==job['scene']]:
            m=StructuredProbe(p['channel_dim'],job['mean'],job['phase_prior'])
            m.load_state_dict(torch.load(record['path'],map_location='cpu',weights_only=True));assert state_hash(m)==record['state_sha256']
            assert sum(v.numel() for v in m.parameters())==q['parameters']
            out=m(x,'uniform');other=m(x,'conditional')
            for key in out:torch.testing.assert_close(out[key],other[key],rtol=0,atol=0)
        torch.manual_seed(7301);m=StructuredProbe(p['channel_dim'],job['mean'],job['phase_prior'])
        with torch.no_grad():m.head.weight.normal_(0,.03);m.query.weight.normal_(0,.01)
        assert not any(isinstance(v,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.Dropout)) for v in m.modules())
        for mode in q['modes']:
            def calc(data,rr=reg,pp=phase):
                output=m(data,mode);selected={k:v[train] for k,v in output.items()}
                loss=probe_loss(selected,rr[train],pp[train],weights)
                grad=torch.autograd.grad(loss,tuple(m.parameters()),allow_unused=True)
                return float(loss.detach()),[None if v is None else v.detach() for v in grad]
            value,grad=calc(x);poison={k:v.clone() for k,v in x.items()}
            for v in poison.values():v[held]=19.25
            rr=reg.clone();rr[held]=9.25;pp=phase.clone();pp[held]=3
            other,g=calc(poison,rr,pp)
            errors=[float((a-b).abs().max()) for a,b in zip(grad,g) if a is not None and b is not None]
            assert all((a is None)==(b is None) for a,b in zip(grad,g))
            diff=max([abs(value-other),*errors]);assert diff<=1e-6
            changed=reg.clone();changed[train[0],0]+=.1;_,gg=calc(x,changed)
            sensitivity=max(float((a-b).abs().max()) for a,b in zip(grad,gg) if a is not None and b is not None);assert sensitivity>1e-6
            if mode=='conditional':assert m.query.weight.grad is None and grad[1] is not None
            if mode=='visual_disabled':
                altered={k:v.clone() for k,v in x.items()};altered['visual'][:]=51;altered['past_visual'][:]=-29
                a,b=m(x,mode),m(altered,mode)
                for key in ('regression','phase'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            checks.append(dict(scene=job['scene'],mode=mode,held_input_label_loss_gradient_error=diff,active_target_gradient_change=sensitivity))
    # Relative-label conversion independently recovers the frozen world targets.
    ee=labels['ee'];r=labels['regression']
    world=np.einsum('ni,nji->nj',r[:,:3]*.1,ee[:,:3,:3])+ee[:,:3,3]
    normals=np.einsum('ni,nji->nj',r[:,3:6],ee[:,:3,:3])
    np.testing.assert_allclose(world,labels['contacts_world'],atol=1e-7,rtol=0)
    np.testing.assert_allclose(normals,labels['normals_world'],atol=1e-7,rtol=0)
    verify(q);verify(p)
    write_json(run/'probe_preflight.json',dict(passed=True,checks=checks,models=27,parameters=q['parameters'],
        same_paired_initial_outputs=True,forward_has_no_privileged_fields=True,canonical_geometry_targets_verified=True,
        source_sha256={str(run/'probe_protocol.json'):digest(run/'probe_protocol.json'),str(Path(__file__)):digest(Path(__file__))}))
    print('PROBE_PREFLIGHT_PASSED',len(checks),q['parameters'],flush=True)


def finalize(run):
    p,q,raw,labels=load_probes(run);result=read(run/'probe_result.json');assert result['status']=='complete'
    assert read(run/'probe_preflight.json')['passed']
    x={k:torch.from_numpy(v) for k,v in raw.items()};summaries=[];seen=set();maxerror=0.
    for record in result['reports']:
        key=(record['scene'],record['mode'],record['seed']);assert key not in seen;seen.add(key)
        job=next(j for j in q['jobs'] if j['scene']==record['scene']);initial=next(i for i in q['initializations'] if (i['scene'],i['seed'])==(record['scene'],record['seed']))
        assert initial['state_sha256']==record['initial_sha256'] and record['updates']==600
        for field,h in (('artifact','sha256'),('checkpoint','checkpoint_sha256')):assert digest(Path(record[field]))==record[h]
        m=StructuredProbe(p['channel_dim'],job['mean'],job['phase_prior']);m.load_state_dict(torch.load(record['checkpoint'],map_location='cpu',weights_only=True));m.eval()
        with torch.no_grad():again={k:v.numpy() for k,v in m(x,record['mode']).items()}
        with np.load(record['artifact']) as z:out={k:z[k].copy() for k in z.files}
        assert set(out)==set(again) and all(np.isfinite(v).all() for v in out.values())
        err=max(float(np.max(np.abs(out[k]-again[k]))) for k in out);assert err<2e-5;maxerror=max(maxerror,err)
        for key in ('current_attention','past_attention'):
            assert np.all(out[key]>=0);np.testing.assert_allclose(out[key].sum(-1),4,atol=2e-6)
            if record['mode']!='conditional':np.testing.assert_array_equal(out[key],np.ones_like(out[key]))
        rec={k:record[k] for k in ('scene','mode','seed')}
        for part in ('train','held'):
            ids=job[part];weights=np.array(job[part+'_weights'])
            rec[part]=metrics(out['regression'][ids],out['phase'][ids],labels['regression'][ids],labels['phase'][ids],weights)
        rec['held']['geometry_gate']=rec['held']['position_cm']<=1 and rec['held']['normal_angle_deg']<=3 and rec['held']['invalid_normals']==0
        rec['attention_deviation']=float(np.max(np.abs(out['current_attention']-1)))
        summaries.append(rec)
    assert seen=={(j['scene'],m,s) for j in q['jobs'] for m in q['modes'] for s in q['seeds']}
    consistent=[]
    for job in q['jobs']:
        scene=job['scene'];prior=next(b['held'] for b in q['baselines'] if b['scene']==scene and b['method']=='training_world_mean')
        comparisons=[]
        for seed in q['seeds']:
            a=next(r for r in summaries if (r['scene'],r['mode'],r['seed'])==(scene,'conditional',seed))['held']
            b=next(r for r in summaries if (r['scene'],r['mode'],r['seed'])==(scene,'uniform',seed))['held']
            comparisons.append(a['position_cm']<b['position_cm'] and a['position_cm']<prior['position_cm'] and a['normal_angle_deg']<=b['normal_angle_deg']+1)
        consistent.append(dict(scene=scene,consistent_improvement=all(comparisons),paired_seed_pass=comparisons))
    verify(q);verify(p)
    write_json(run/'probe_audit.json',dict(passed=True,models=len(seen),cpu_prediction_max_error=maxerror,
        source_sha256={str(run/'probe_result.json'):digest(run/'probe_result.json'),str(Path(__file__)):digest(Path(__file__))}))
    write_json(run/'probe_summary.json',dict(status='complete',reports=summaries,baselines=q['baselines'],consistent_scene_improvement=consistent,
        development_signal=sum(r['consistent_improvement'] for r in consistent)>=2,
        scope='Three developer scenes only, geometry/stage diagnostic head. Not an action adapter or task-success result.',
        source_sha256={str(run/'probe_protocol.json'):digest(run/'probe_protocol.json'),str(run/'probe_result.json'):digest(run/'probe_result.json'),
            str(run/'probe_audit.json'):digest(run/'probe_audit.json')}))
    lines=['# S2g structured readout: final600, all3seeds','',
        'Position is relative functional-contact-point error; normal is relative object-facing normal. Teacher stage is a diagnostic target only.',
        '', '|Held scene|Readout|Position cm (seed mean / min–max)|Normal degrees|Phase accuracy|Grip MAE|Geometry gates|',
        '|---|---|---|---|---|---|---|']
    for job in q['jobs']:
        for mode in q['modes']:
            selected=[r for r in summaries if r['scene']==job['scene'] and r['mode']==mode];mm=[r['held'] for r in selected]
            pos=[v['position_cm'] for v in mm];angle=np.mean([v['normal_angle_deg'] for v in mm]);phase=np.mean([v['phase_accuracy'] for v in mm]);grip=np.mean([v['gripper_mae'] for v in mm])
            lines.append(f"|{job['scene']}|{mode}|{np.mean(pos):.2f} / {min(pos):.2f}–{max(pos):.2f}|{angle:.2f}|{phase:.3f}|{grip:.4f}|{sum(v['geometry_gate'] for v in mm)}/3|")
    lines+=['','Training-only world-mean baseline and same-target analytic PCA/structured controls:','',
        '|Held scene|Baseline|Position cm|Normal degrees|Phase accuracy|','|---|---|---|---|---|']
    for b in q['baselines']:
        m=b['held'];lines.append(f"|{b['scene']}|{b['method']}|{m['position_cm']:.2f}|{m['normal_angle_deg']:.2f}|{m['phase_accuracy']:.3f}|")
    lines+=['',f'Prespecified development signal (consistent gain on≥2scenes): {sum(r["consistent_improvement"] for r in consistent)>=2}.',
        'All models retained. No current action head, cap, base noise, or simulation changed.']
    (run/'probe_summary.md').write_text('\n'.join(lines)+'\n');print('PROBE_AUDIT_SUMMARY_COMPLETE',maxerror,consistent,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);(preflight if args.preflight else finalize)(args.run.resolve())
