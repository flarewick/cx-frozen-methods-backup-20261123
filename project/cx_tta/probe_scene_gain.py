"""Fit fixed small gain/risk probes on scene-OOF candidates; calibrate separately."""
import argparse
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from .paths import ROOT, write_json, replace_json
from .audit import digest
from .probe_candidates import read, verify
from .scene_gain_probe import (observable_features, labels, row_weights, GainProbe, select,
                               selected_metrics, calibrate, chosen_actions)
from .observable_candidate_scores import weighted_auc
from .robot_kinematics import RobotFK
from .candidate_diagnostic import row_metrics, summarize


def load_inputs(run):
    p, result = read(run/'protocol.json'), read(run/'result.json')
    verify(p)
    if not read(run/'candidate_audit.json')['passed']:
        raise ValueError('Candidate audit required')
    with np.load(run/'data.npz') as z:
        data = {k:z[k].copy() for k in ('base','target','valid')}
    outer = next(x for x in p['jobs'] if x['name']=='outer')
    with np.load(outer['context']) as z:
        context = z['context'].copy()
    proposals = {encoding:np.zeros((3,*data['base'].shape),np.float32) for encoding in p['encodings']}
    coverage = {encoding:np.zeros((3,len(p['rows'])),int) for encoding in p['encodings']}
    for report in result['reports']:
        if digest(Path(report['file']))!=report['sha256']:
            raise ValueError('Changed candidate prediction')
        with np.load(report['file']) as z:
            ids, pred = z['indices'],z['predicted']
        proposals[report['encoding']][report['seed'],ids] = pred
        coverage[report['encoding']][report['seed'],ids] += 1
    if not all(np.all(v==1) for v in coverage.values()):
        raise ValueError('OOF candidate cover must be exact once per row and initialization')
    return p, data, context, proposals


def prepare(run):
    p,data,context,proposals=load_inputs(run)
    files=[run/f for f in ('protocol.json','result.json','candidate_audit.json','preflight_audit.json','data.npz')]
    files += [Path(__file__),ROOT/'cx_tta/scene_gain_probe.py',ROOT/'scripts/probe_scene_gain.sh']
    files += [Path(x['file']) for x in read(run/'result.json')['reports']]
    write_json(run/'probe_protocol.json',dict(status='frozen_before_labels_and_probe_fitting',design=p['probe_design'],
        source_sha256={str(f):digest(f) for f in files},
        loss='SmoothL1(gain,clip((Lbase-Lcandidate)/(1+Lbase),-5,1)) + BCE(strict benefit) + BCE(tolerant harm); '
             'AdamW default decay.01, lr.001,600 updates,512 samples/update, fixed balanced draws. '
             'Gain scales: arm6/7 native BC+rotation+position; gripMSE/.2^2. No positive-class reweighting.',
        input_dim=234,scaler='Weighted mean/std fitted on gate-fit OOF features only, std floor.05. '
            'Common outer PCA coordinate for ALL router inputs, candidate fold PCA never concatenated.',
        scope=p['scope']))
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    manifest=[]
    for encoding in p['encodings']:
        for support in ('raw','capped'):
            x,scores=observable_features(context,data['base'],proposals[encoding],support)
            if x.shape!=(3,len(p['rows']),2,2,3,234) or not np.isfinite(x).all():
                raise ValueError('Observable feature shape/finite failure')
            panel=labels(data['base'],data['target'],data['valid'],proposals[encoding],support,robots)
            if any(np.any(v[...,0]) for v in panel.values()):
                raise ValueError('Base action must have zero benefit, gain and harm')
            file=run/f'probe_labels_{encoding}_{support}.npz'
            np.savez_compressed(file,**panel)
            manifest.append(dict(encoding=encoding,support=support,file=str(file),sha256=digest(file)))
            print('LABELS',encoding,support,flush=True)
    verify(read(run/'probe_protocol.json'))
    write_json(run/'probe_data_manifest.json',dict(status='complete',records=manifest))


def rank_report(prediction,panel,proxies,rows,ids):
    report={}
    for component,name in enumerate(('arm','grip')):
        weights=row_weights(rows,ids,'expert')
        mass=np.broadcast_to(weights[None,:,None,None],(3,len(rows),2,3)).ravel()
        y=panel['benefit'][:,:,:,component,1:].ravel()
        h=panel['harm'][:,:,:,component,1:].ravel()
        use=mass>0
        if not use.any():
            continue
        w=mass[use]
        report[name]=dict(benefit_auc=weighted_auc(prediction[:,:,:,component,:,1].ravel()[use],y[use],w),
            gain_auc=weighted_auc(prediction[:,:,:,component,:,0].ravel()[use],y[use],w),
            harm_auc=weighted_auc(prediction[:,:,:,component,:,2].ravel()[use],h[use],w),
            benefit_prevalence=float(np.average(y[use],weights=w)),harm_prevalence=float(np.average(h[use],weights=w)),
            proxies={k:weighted_auc(v[:,:,:,component].ravel()[use],y[use],w) for k,v in proxies.items()})
    return report


def selected_reports(panel,choices,rows,ids):
    out={}
    for kind in ('expert','anchor','recovery'):
        weights=row_weights(rows,ids,kind)
        if not weights.sum():continue
        out[kind]=dict(pooled=selected_metrics(panel,choices,weights),
            candidate_seeds=[selected_metrics({k:v[s:s+1] for k,v in panel.items()},choices[s:s+1],weights) for s in range(3)],
            per_task={t:selected_metrics(panel,choices,row_weights(rows,[i for i in ids if rows[i]['task']==t],kind))
                      for t in sorted({rows[i]['task'] for i in ids if rows[i]['kind']==kind})})
    return out


def validation_geometry(data,proposals,choices,support,rows,ids,robots):
    ix=np.array(ids)
    base,target,valid=(data[k][ix] for k in ('base','target','valid'))
    selected_rows=[rows[i] for i in ix]
    baseline=row_metrics(base.astype(float),target,valid,robots)
    results=[]
    for seed in range(3):
        action=chosen_actions(base,proposals[seed,ix],choices[seed,ix],support)
        metric=row_metrics(action,target,valid,robots)
        parts={}
        for kind in ('expert','anchor'):
            part=[i for i,r in enumerate(selected_rows) if r['kind']==kind]
            parts[kind]=summarize(metric,action,target,valid,selected_rows,part,baseline)
        results.append(dict(candidate_seed=seed,**parts))
    return results


def train(run,device):
    cp=read(run/'probe_protocol.json');verify(cp)
    p,data,context,proposals=load_inputs(run)
    if (run/'probe_result.json').exists() or (run/'probe_environment.json').exists():
        raise ValueError('Fresh probe training required')
    cfg=p['probe_design'];reports=[];start=time.monotonic()
    write_json(run/'probe_environment.json',dict(device=device,torch=torch.__version__,
        gpu=None if device=='cpu' else torch.cuda.get_device_name(0)))
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')]
    baselines=[]
    try:
        for entry in read(run/'probe_data_manifest.json')['records']:
            encoding,support=entry['encoding'],entry['support']
            if digest(Path(entry['file']))!=entry['sha256']:raise ValueError('Changed frozen probe labels')
            with np.load(entry['file']) as z:panel={k:z[k].copy() for k in z.files}
            inputs,proxies=observable_features(context,data['base'],proposals[encoding],support)
            features=inputs.reshape(-1,inputs.shape[-1])
            rweights=row_weights(p['rows'],p['parts']['gate_fit'])
            mass=np.broadcast_to(rweights[None,:,None,None,None],inputs.shape[:-1]).ravel().copy()
            mass/=mass.sum()
            mean=np.average(features,axis=0,weights=mass).astype(np.float32)
            std=np.sqrt(np.average((features-mean)**2,axis=0,weights=mass)).clip(.05).astype(np.float32)
            x=torch.from_numpy(((features-mean)/std).astype(np.float32)).to(device)
            targets=np.stack([np.clip(panel['gain'][...,1:],-5,1),panel['benefit'][...,1:],panel['harm'][...,1:]],-1).reshape(-1,3).astype(np.float32)
            y=torch.from_numpy(targets).to(device)
            for architecture in cfg['models']:
                for seed in cfg['seeds']:
                    torch.manual_seed(seed)
                    model=GainProbe(features.shape[-1],architecture).to(device)
                    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
                    generator=torch.Generator().manual_seed(74219+seed)
                    draws=torch.multinomial(torch.from_numpy(mass),cfg['updates']*cfg['batch_size'],replacement=True,generator=generator).reshape(cfg['updates'],cfg['batch_size'])
                    history=[]
                    for update,index in enumerate(draws,1):
                        index=index.to(device);opt.zero_grad();raw=model(x[index]);truth=y[index]
                        loss=F.smooth_l1_loss(raw[:,0],truth[:,0])+F.binary_cross_entropy_with_logits(raw[:,1:],truth[:,1:]) * 2
                        if not torch.isfinite(loss):raise ValueError('Nonfinite probe loss')
                        loss.backward();opt.step()
                        if update==1 or update%100==0:history.append(dict(update=update,loss=float(loss.detach())))
                    tag=f'probe_{encoding}_{support}_{architecture}_s{seed}'
                    checkpoint=run/(tag+'.pt');torch.save(model.state_dict(),checkpoint)
                    restored=GainProbe(features.shape[-1],architecture).to(device)
                    restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
                    model.eval();restored.eval();values=[]
                    with torch.no_grad():
                        for offset in range(0,len(x),512):
                            raw=model(x[offset:offset+512])
                            if not torch.equal(raw,restored(x[offset:offset+512])):raise ValueError('Probe reload mismatch')
                            raw[:,1:]=raw[:,1:].sigmoid();values.append(raw.cpu().numpy())
                    predicted=np.concatenate(values).reshape(*inputs.shape[:-1],3)
                    best,trials=calibrate(predicted,panel,p['rows'],p['parts']['calibration'])
                    choices=np.zeros(predicted.shape[:-2],np.int16) if best['gain_threshold'] is None else select(predicted,best['gain_threshold'],best['risk_threshold'])
                    artifact=run/(tag+'.npz')
                    np.savez_compressed(artifact,prediction=predicted,choices=choices,mean=mean,std=std,
                        sampled_counts=np.bincount(draws.numpy().ravel(),minlength=len(features)))
                    write_json(run/(tag+'_history.json'),history)
                    reports.append(dict(encoding=encoding,support=support,architecture=architecture,seed=seed,
                        parameters=sum(v.numel() for v in model.parameters()),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                        artifact=str(artifact),sha256=digest(artifact),calibration=best,calibration_trials=trials,
                        validation_rank=rank_report(predicted,panel,proxies,p['rows'],p['parts']['validation']),
                        validation_selected=selected_reports(panel,choices,p['rows'],p['parts']['validation']),
                        validation_geometry=validation_geometry(data,proposals[encoding],choices,support,p['rows'],p['parts']['validation'],robots)))
                    replace_json(run/'probe_progress.json',dict(completed=len(reports),tag=tag,elapsed_seconds=time.monotonic()-start))
                    print('PROBE',tag,'calibration',best['gain_threshold'],best['risk_threshold'],flush=True)
            for alpha in range(4):
                choice=np.full((3,len(p['rows']),2,2),alpha,np.int16)
                baselines.append(dict(encoding=encoding,support=support,rule=f'fixed_{(0,.25,.5,1)[alpha]}',
                    validation_selected=selected_reports(panel,choice,p['rows'],p['parts']['validation'])))
            # Label-known potential only; choose actual gain under strict per-component benefit.
            potential=np.where(panel['benefit'],panel['gain'],-np.inf)
            potential[...,0]=0
            choice=potential.argmax(-1)
            baselines.append(dict(encoding=encoding,support=support,rule='teacher_known_strict_component_potential',
                validation_selected=selected_reports(panel,choice,p['rows'],p['parts']['validation'])))
        verify(cp);verify(p)
        write_json(run/'probe_result.json',dict(status='complete',reports=reports,baselines=baselines,
            elapsed_seconds=time.monotonic()-start,promote=False,tta=False,scope=cp['scope']))
        replace_json(run/'probe_progress.json',dict(phase='complete',models=len(reports)))
        print('PROBES_COMPLETE',run,flush=True)
    except BaseException:
        write_json(run/'probe_failure.json',dict(traceback=traceback.format_exc(),completed_models=len(reports)))
        raise


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args();torch.set_num_threads(2);threadpool_limits(2);torch.use_deterministic_algorithms(True)
    if args.prepare_only:prepare(args.run.resolve())
    else:train(args.run.resolve(),args.device)


if __name__=='__main__':main()
