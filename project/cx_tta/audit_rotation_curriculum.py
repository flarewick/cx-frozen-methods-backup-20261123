"""Independent source/sampling/float64 pose audit and fixed curriculum gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .audit import digest
from .audit_pose_capacity import numpy_fk,angles,episode_metrics
from .paths import write_json
from .pose_adapter import PoseAdapter
from .robot_kinematics import RobotFK
from .train_semantic_pilot import balanced_weights

def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();protocol=read(run/'protocol.json');result=read(run/'result.json');rows=protocol['rows'];torch.set_num_threads(2)
    if result['status']!='complete' or len(result['reports'])!=9:raise ValueError('Nine completed new variants required')
    for f,sha in protocol['source_sha256'].items():
        if digest(Path(f))!=sha:raise ValueError('Changed source '+f)
    for reused in protocol['reused']:
        for f,sha in reused['sha256'].items():
            if digest(Path(f))!=sha:raise ValueError('Changed reused control '+f)
    if digest(run/'sampling.npz')!=protocol['sampling_sha256']:raise ValueError('Changed preregistration')
    with np.load(run/'sampling.npz') as z:sampling={k:z[k].copy() for k in z.files}
    prepared=Path(protocol['prepared'])
    with np.load(prepared/'data.npz') as z:data={k:z[k].copy() for k in z.files}
    config=read(Path(protocol['config']));bots=[RobotFK(config,s) for s in ('left','right')];parts={s:np.array([i for i,r in enumerate(rows) if r['pilot_split']==s]) for s in ('fit','inner_held','anchor_fit','anchor_held')};fit=parts['fit'];anchors=parts['anchor_fit']
    if not np.array_equal(sampling['fit_indices'],fit):raise ValueError('Fit partition mismatch')
    for first,second in [('fit','inner_held'),('anchor_fit','anchor_held')]:
        if {rows[i]['group'] for i in parts[first]}&{rows[i]['group'] for i in parts[second]}:raise ValueError('Group leak')
    def endpoints(x):return np.stack([numpy_fk(r,x[:,protocol['offsets'],start:start+6].astype(float)) for r,start in zip(bots,(0,7))],2)
    ba=angles(endpoints(data['base'][fit]),endpoints(data['target'][fit]));valid=np.broadcast_to(data['valid'][fit][:,protocol['offsets'],None],ba.shape);rare=np.any(valid&(ba>=45),(1,2))
    if not np.array_equal(rare,sampling['rare_rows']) or np.max(np.abs(ba-sampling['fit_base_angles']))>1e-9:raise ValueError('Demand labels changed')
    groups=sorted({rows[i]['group'] for i,v in zip(fit,rare) if v});counts={g:sum(v and rows[i]['group']==g for i,v in zip(fit,rare)) for g in groups};demand=np.array([1/(len(groups)*counts[rows[i]['group']]) if v else 0 for i,v in zip(fit,rare)]);normal=balanced_weights([rows[i] for i in fit]).numpy();normal/=normal.sum()
    if not np.array_equal(demand,sampling['demand']) or not np.allclose(.5*normal+.5*demand,sampling['weights'],atol=1e-15):raise ValueError('Sampler not preregistered mixture')
    files=[run/'protocol.json',run/'result.json',run/'environment.json',run/'sampling.npz',Path(__file__)];metrics=[];episode_rows=[];parity=[];draw_records=[]
    for report in result['reports']:
        seed,variant=report['seed'],report['variant'];torch.manual_seed(seed);model=PoseAdapter(protocol['context_dim'],proposal=True);h=hashlib.sha256()
        for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
        if h.hexdigest()!=report['initial_state_sha256'] or report['parameters']!=sum(v.numel() for v in model.parameters()):raise ValueError('Initial/parameter mismatch')
        file=run/f'{variant}_s{seed}_draws.npy';files.append(file);draws=np.load(file);gen=torch.Generator().manual_seed(seed);expected=np.stack([fit[torch.multinomial(torch.from_numpy(sampling['weights']),len(fit),replacement=True,generator=gen).numpy()] for _ in range(protocol['epochs'])])
        if not np.array_equal(draws,expected) or report['updates']!=140 or not np.isin(draws,fit).all():raise ValueError('Draw/update/partition mismatch')
        local={v:i for i,v in enumerate(fit)};rare_count=sum(rare[local[int(i)]] for i in draws.ravel());draw_records.append(dict(seed=seed,variant=variant,draws=draws.size,rare_draws=int(rare_count),rare_fraction=float(rare_count/draws.size)))
        if variant.endswith('_preserve'):
            file=run/f'{variant}_s{seed}_anchor_draws.npy';files.append(file);adraws=np.load(file);gen=torch.Generator().manual_seed(seed+123);aw=balanced_weights([rows[i] for i in anchors]);expected=[]
            for epoch in range(protocol['epochs']):
                values=[]
                for start in range(0,len(fit),64):values.extend(anchors[torch.multinomial(aw,min(64,len(fit)-start),replacement=True,generator=gen).numpy()].tolist())
                expected.append(values)
            if not np.array_equal(adraws,expected) or not np.isin(adraws,anchors).all():raise ValueError('Anchor sampling leak/mismatch')
        checkpoint=run/f'{variant}_s{seed}.pt';files.extend([checkpoint,run/f'{variant}_s{seed}_history.json'])
        if digest(checkpoint)!=report['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        for part,expected_indices in parts.items():
            path=run/f'{variant}_s{seed}_{part}.npz';files.append(path);record=next(f for f in report['files'] if f['file']==str(path))
            if digest(path)!=record['sha256']:raise ValueError('Predictions changed')
            with np.load(path) as z:saved={k:z[k].copy() for k in z.files}
            ids=saved['indices']
            if not np.array_equal(ids,expected_indices) or not all(np.array_equal(saved[k],data[k][ids]) for k in ('base','target','valid')):raise ValueError('Saved data mismatch')
            pp,bp,tp=[endpoints(saved[k]) for k in ('predicted','base','target')];angle,baseangle=angles(pp,tp),angles(bp,tp);translation=np.linalg.norm(pp[...,:3,3]-tp[...,:3,3],axis=-1);mask=np.broadcast_to(saved['valid'][:,protocol['offsets'],None],angle.shape);large=mask&(baseangle>=45)
            diff=float(np.max(np.abs(angle-saved['angle'])))
            if diff>.06 or np.max(np.abs(translation-saved['translation']))>2e-6 or not np.array_equal(mask,saved['mask']) or not np.array_equal(large,saved['large']):raise ValueError('Independent pose/mask mismatch')
            ms,erows=episode_metrics(angle,baseangle,translation,mask,large,ids,rows,saved['predicted'],saved['target'],saved['valid'])
            for task,values in ms.items():
                for k,value in values.items():
                    actual=report['metrics'][part][task][k]
                    if (value is None)!=(actual is None) or (value is not None and abs(value-actual)>(.06 if 'angle' in k or k=='bounded_lower_bound_deg' else 1e-5)):raise ValueError('Aggregate mismatch '+k)
            metrics.append(dict(seed=seed,variant=variant,partition=part,metrics=ms));episode_rows.extend(dict(seed=seed,variant=variant,partition=part,**r) for r in erows);parity.append(dict(seed=seed,variant=variant,partition=part,max_angle_difference_deg=diff))
    parent=Path(protocol['parent']);old=read(parent/'independent_metrics.json');oldfit=read(parent/'fit_capacity.json')
    for reuse in protocol['reused']:
        seed,pv,variant=reuse['seed'],reuse['parent_variant'],reuse['variant']
        for r in old['records']:
            if r['seed']==seed and r['variant']==pv:metrics.append({**r,'variant':variant,'reused_from':str(parent)})
        r=next(r for r in oldfit['reports'] if r['seed']==seed and r['variant']==pv);metrics.append(dict(seed=seed,variant=variant,partition='fit',metrics=r['metrics'],reused_from=str(parent)))
    write_json(run/'independent_metrics.json',dict(records=metrics,episodes=episode_rows))
    summary=[]
    for variant in ('uniform_bc','uniform_pose',*protocol['variants']):
        table=dict(variant=variant)
        for part in ('fit','inner_held','anchor_held'):
            rr=[r for r in metrics if r['variant']==variant and r['partition']==part];table[part]={}
            for task in rr[0]['metrics']:
                table[part][task]={k:None if rr[0]['metrics'][task][k] is None else float(np.mean([r['metrics'][task][k] for r in rr])) for k in ('base_angle_deg','angle_deg','translation_m','large_angle_deg','large_under_15deg_fraction','native_action_mse')}
        summary.append(table)
    gates=protocol['gates'];gate_rows=[]
    for variant in protocol['variants']:
        rr=[r for r in metrics if r['variant']==variant];fitpass=heldpass=anchorpass=True;detail=[]
        for r in rr:
            for task,m in r['metrics'].items():
                if r['partition']=='fit' and m['large_angle_deg'] is not None:
                    ok=m['large_angle_deg']<=m['large_base_angle_deg']*(1-gates['fit_large_mean_reduction_min']) and m['large_under_15deg_fraction']>=gates['fit_large_fraction_under_15deg_min'];fitpass&=ok;detail.append(dict(seed=r['seed'],task=task,gate='fit_large',passed=bool(ok)))
                if r['partition']=='inner_held' and m['large_angle_deg'] is not None:
                    ok=m['large_angle_deg']<=m['large_base_angle_deg']*(1-gates['held_large_mean_reduction_min']);heldpass&=ok;detail.append(dict(seed=r['seed'],task=task,gate='held_large',passed=bool(ok)))
                if r['partition']=='anchor_held':anchorpass&=m['angle_deg']<=gates['anchor_held_angle_max_deg']
        preservepass=None
        if variant.endswith('_preserve'):
            preservepass=True
            for r in [r for r in rr if r['partition']=='anchor_held']:
                control=next(x for x in metrics if x['seed']==r['seed'] and x['variant']=='demand_pose' and x['partition']=='anchor_held')
                for task,m in r['metrics'].items():preservepass&=m['native_action_mse']<=gates['preserve_anchor_mse_ratio_max']*control['metrics'][task]['native_action_mse']
        gate_rows.append(dict(variant=variant,fit_large_passed=bool(fitpass),held_large_passed=bool(heldpass),anchor_angle_passed=bool(anchorpass),preservation_ratio_passed=None if preservepass is None else bool(preservepass),details=detail))
    write_json(run/'audit.json',dict(passed=True,new_variants=9,reused_variants=6,parameters=40718,updates_per_variant=140,draw_records=draw_records,fk_parity=parity,checks=['Pre-training protocol and source hashes','Fit-only demand mixture independently reconstructed','Every actual expert/anchor draw matches fixed independent generators','Identical model initialization and expert draw order across new variants','No fit/held/anchor group leakage','Exact cached data and final checkpoint identities','Float64 independent FK/valid/large-mask/episode metrics','Six old uniform controls reused, no retraining'],source_sha256={str(f):digest(f) for f in files},scope=protocol['scope']))
    write_json(run/'decision.json',dict(status='do_not_promote',summary=summary,gates=gate_rows,interpretation='Oversampling helps training-inner rotation learning but does not establish held recovery. Successful-action penalty does not automatically preserve unseen success scenes. Single held large-angle scene cannot support broad generalization; no closed-loop claim.',next='No bulk data/TTA/deployment. Separate fit capacity/budget/representation from recovery-state coverage using existing real inputs, and inventory executed recovery labels before missing-interface caching.',source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'audit.json',run/'independent_metrics.json',Path(__file__)]}))
    print(json.dumps(dict(audit_passed=True,summary=summary,gates=[{k:v for k,v in g.items() if k!='details'} for g in gate_rows]),indent=2))

if __name__=='__main__':main()
