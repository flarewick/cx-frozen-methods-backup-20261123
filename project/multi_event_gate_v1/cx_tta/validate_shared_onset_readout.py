"""Frozen CPU pilot: scene-grouped OOF, then one independent calibration pass.

Offline selection among actual paired Z/F outcomes is not a new closed-loop
success result. No threshold search, GT forward fields, VLA calls or TTA.
"""
import argparse
import copy
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT,output_path,new_run,write_json,replace_json
from .native_history_cache import read,verify,require,arrays
from .audit import digest
from .onset_decision_dataset import CLASSES,require_fit_support,support
from .onset_observables import validate_packet
from .candidate_benefit_readout import fit_readout,CandidateBenefitReadout,GroupProjection

VARIANTS={'observation_base_four_class':False,'candidate_conditioned_four_class':True}


def fold_view(rows,held_group=None):
    """Transient fitter view only; original split rows on disk never change.

    The outer calibration metadata supplies the already audited global support
    check; all its packets, plus held fit packets, are inaccessible to fitting.
    Its labels are not gradient targets, model-selection scores or thresholds.
    """
    view=copy.deepcopy(rows)
    held=[]
    if held_group is not None:
        held=[r['index'] for r in rows if r['split']=='fit' and r['group']==held_group]
        require(bool(held),'Whole fit group exists')
        for i in held:view[i]['split']='calibration'
    train=[r['index'] for r in view if r['split']=='fit']
    require(not {rows[i]['group'] for i in train}&{rows[i]['group'] for i in held},'No held scene in fit')
    return view,train,held


def summarize_selection(rows,decisions):
    require(set(decisions)=={r['index'] for r in rows},'One decision per assessed row')
    require(all(type(v)is bool for v in decisions.values()),'Boolean selection')
    z=sum(r['Z_success'] for r in rows);full=sum(r['F_success'] for r in rows)
    selected=sum(decisions[r['index']] for r in rows)
    rescued=sum(decisions[r['index']] and r['label']=='rescue' for r in rows)
    harmed=sum(decisions[r['index']] and r['label']=='harm' for r in rows)
    success=sum(r['F_success'] if decisions[r['index']] else r['Z_success'] for r in rows)
    require(success==z+rescued-harmed,'Paired value accounting')
    return dict(rows=len(rows),scenes=len({r['group'] for r in rows}),successes=success,
        baseline_successes=z,always_F_successes=full,selected=selected,rejected=len(rows)-selected,
        rescued=rescued,harmed=harmed,net=rescued-harmed,base_success_preserved=z-harmed,
        base_success_denominator=z,missed_rescues=sum(r['label']=='rescue' for r in rows)-rescued,
        rejected_harms=sum(r['label']=='harm' for r in rows)-harmed,
        risk_weighted_utility=rescued-2*harmed,
        by_task={t:_task_summary([r for r in rows if r['task']==t],decisions) for t in sorted({r['task'] for r in rows})})


def _task_summary(rows,decisions):
    result=summarize_selection_no_tasks(rows,decisions)
    return result


def summarize_selection_no_tasks(rows,decisions):
    rescued=sum(decisions[r['index']] and r['label']=='rescue' for r in rows)
    harmed=sum(decisions[r['index']] and r['label']=='harm' for r in rows)
    return dict(rows=len(rows),scenes=len({r['group'] for r in rows}),
        baseline_successes=sum(r['Z_success'] for r in rows),always_F_successes=sum(r['F_success'] for r in rows),
        successes=sum(r['F_success'] if decisions[r['index']] else r['Z_success'] for r in rows),
        selected=sum(decisions[r['index']] for r in rows),rescued=rescued,harmed=harmed,net=rescued-harmed)


def task_constant(training_rows,task,cost):
    rr=[r for r in training_rows if r['task']==task];require(bool(rr),'Task baseline has training support')
    counts=Counter(r['group'] for r in rr)
    utility=sum(((r['label']=='rescue')-cost*(r['label']=='harm'))/counts[r['group']] for r in rr)
    return bool(utility>0)  # Base on exact tie; no holdout outcome is consulted.


def prediction_rows(model,packets,indices,rows):
    output=[]
    for i in indices:
        p=model.predict(packets[i]);scores=p['class_scores']
        output.append(dict(index=i,group=rows[i]['group'],task=rows[i]['task'],block=rows[i]['block'],
            scores=scores,calibrated=False,
            decisions={str(c):bool(scores['rescue']-c*scores['harm']>0) for c in (0,2)}))
    return output


def save_model(folder,model):
    folder.mkdir()
    values={'weights':model.weights,'bias':model.bias}
    groups=list(model.projections)
    for i,k in enumerate(groups):
        for field in ('mean','scale','basis','projected_scale'):values[f'g{i}_{field}']=getattr(model.projections[k],field)
    np.savez_compressed(folder/'model.npz',**values)
    write_json(folder/'metadata.json',dict(model.metadata,groups=groups,model_sha256=digest(folder/'model.npz')))


def load_model(folder):
    m=read(folder/'metadata.json');require(digest(folder/'model.npz')==m['model_sha256'],'Checkpoint integrity')
    z=arrays(folder/'model.npz');projections={}
    for i,k in enumerate(m['groups']):projections[k]=GroupProjection(**{f:z[f'g{i}_{f}'] for f in ('mean','scale','basis','projected_scale')})
    return CandidateBenefitReadout(projections,z['weights'],z['bias'],m)


def assess(rows,predictions,controls):
    result={'controls':{k:summarize_selection(rows,v) for k,v in controls.items()},'models':{}}
    for name,preds in predictions.items():
        lookup={r['index']:r for r in preds};require(len(lookup)==len(preds),'Unique OOF/calibration predictions')
        complete=set(lookup)=={r['index'] for r in rows}
        # Never silently discard unsupported folds from overall totals.
        if not complete:
            result['models'][name]={'status':'unestimable','missing_indices':sorted({r['index'] for r in rows}-set(lookup))};continue
        scored=np.asarray([[lookup[r['index']]['scores'][c] for c in CLASSES] for r in rows])
        labels=np.asarray([CLASSES.index(r['label']) for r in rows]);onehot=np.eye(4)[labels]
        result['models'][name]=dict(status='complete',
            cross_entropy=float(-np.log(np.maximum(scored[np.arange(len(rows)),labels],1e-12)).mean()),
            brier=float(np.square(scored-onehot).sum(1).mean()),
            policies={str(c):summarize_selection(rows,{i:lookup[i]['decisions'][str(c)] for i in lookup}) for c in (0,2)})
    return result


def progression(oof,calibration):
    reasons=[]
    for stage,s in [('OOF',oof),('calibration',calibration)]:
        m=s['models']['candidate_conditioned_four_class']
        if m['status']!='complete':reasons.append(stage+': missing folds');continue
        primary=m['policies']['2'];f=s['controls']['always_F'];tc=s['controls']['task_constant_cost2']
        if primary['net']<=0:reasons.append(stage+': primary has no positive net over base')
        if primary['successes']<=f['successes']:reasons.append(stage+': primary does not exceed unconditional F')
        if primary['harmed']>=f['harmed']:reasons.append(stage+': primary does not reduce observed harms')
        if primary['successes']<=tc['successes']:reasons.append(stage+': primary does not exceed fit-only task constant')
        if any(v['net']<0 for v in primary['by_task'].values()):reasons.append(stage+': a task loses successes')
    return dict(signal_gate_passed=not reasons,reasons=reasons,closed_loop_started=False,
        scope='Conservative pilot screening, not a statistical safety certificate; no automatic GPU rollout')


def execute(protocol):
    p=read(protocol);verify(p['source_sha256']);require(p['coverage_passed'] and p['stage']=='shared_onset_readout_cpu_pilot','Frozen pilot identity')
    require(p['model']==dict(rank_per_group=4,steps=200,learning_rate=.03,l2=.1,initialization='exact_zero',device='CPU',scene_equal_loss_weight=True),'Fixed tested budget')
    require(p['comparisons']==list(VARIANTS) and p['fixed_score_ablations']==dict(harm_costs=[0,2],score_margin=0), 'Fixed input and score ablations')
    require(p['primary_selection']==dict(readout='candidate_conditioned_four_class',harm_cost=2,score_margin=0,rule='rescue_score - 2*harm_score >0'), 'Frozen primary rule')
    data=Path(p['data']);m=read(data/'manifest.json');verify(m['source_sha256'])
    require(digest(data/'observables.npz')==m['observables_sha256'] and digest(data/'offline_index.json')==m['offline_index_sha256'],'Data identity')
    rows=read(data/'offline_index.json')['rows'];require_fit_support(rows,p['support_gate'])
    fit=[r['index'] for r in rows if r['split']=='fit'];cal=[r['index'] for r in rows if r['split']=='calibration']
    require(fit==p['fit_rows'] and cal==p['calibration_rows'],'Original split indices')
    out=new_run('shared_onset',group=('effect_learning',));print('RUN',out,flush=True)
    write_json(out/'protocol.json',dict(parent_protocol=str(protocol),parent_sha256=digest(protocol),recipe=p,
        driver_sha256=digest(Path(__file__)),training_authorized_by_user=True,
        progression_gate='Primary FULL cost2: complete predictions; positive net vs base; exceeds F and fit-only task constant; fewer harms than F; no task loses successes, in both OOF and calibration',
        tie_policy='base',calibration_not_opened_for_prediction=True,fold_models_saved=False))
    try:
        z=arrays(data/'observables.npz')
        # Only fit packets instantiated before both models and OOF are finalized.
        packets=[None]*len(rows)
        for i in fit:packets[i]=validate_packet({k:v[i:i+1] for k,v in z.items()})
        kwargs=dict(rank=p['model']['rank_per_group'],steps=p['model']['steps'],learning_rate=p['model']['learning_rate'],l2=p['model']['l2'])
        groups=sorted({rows[i]['group'] for i in fit});predictions={k:[] for k in VARIANTS};folds=[]
        controls={'always_base':{i:False for i in fit},'always_F':{i:True for i in fit},'task_constant_cost0':{},'task_constant_cost2':{}}
        for g in groups:
            view,train,held=fold_view(rows,g);sup=support(view,p['support_gate'])
            fold=dict(group=g,training_indices=train,held_indices=held,support=sup,models={})
            training=[rows[i] for i in train]
            for i in held:
                for cost in (0,2):controls['task_constant_cost'+str(cost)][i]=task_constant(training,rows[i]['task'],cost)
            if sup['coverage_passed']:
                guarded=[packets[i] if i in train else None for i in range(len(rows))]
                for name,include in VARIANTS.items():
                    model=fit_readout(guarded,view,p['support_gate'],include_candidate=include,**kwargs)
                    require(model.metadata['fit_indices']==train,'Only remaining fit scenes used')
                    predictions[name]+=prediction_rows(model,packets,held,rows);fold['models'][name]=model.metadata
            else:fold['status']='unestimable_support'
            folds.append(fold);replace_json(out/'progress.json',dict(stage='whole_scene_OOF',completed=len(folds),requested=len(groups)))
            print('OOF',len(folds),'/',len(groups),g,'support',sup['coverage_passed'],flush=True)
        oof=assess([rows[i] for i in fit],predictions,controls)
        write_json(out/'oof_predictions.json',predictions);write_json(out/'folds.json',folds);write_json(out/'oof_metrics.json',oof)
        models={};training_predictions={}
        for name,include in VARIANTS.items():
            model=fit_readout(packets,rows,p['support_gate'],include_candidate=include,**kwargs)
            require(model.metadata['fit_indices']==fit,'Only full fit subset used')
            save_model(out/name,model);reloaded=load_model(out/name)
            require(all(model.predict(packets[i])==reloaded.predict(packets[i]) for i in fit),'Numeric reload reproduces fit scores')
            models[name]=reloaded;training_predictions[name]=prediction_rows(reloaded,packets,fit,rows)
        full_controls={'always_base':{i:False for i in fit},'always_F':{i:True for i in fit}}
        for cost in (0,2):full_controls['task_constant_cost'+str(cost)]={i:task_constant([rows[j] for j in fit],rows[i]['task'],cost) for i in fit}
        write_json(out/'fit_predictions.json',training_predictions);write_json(out/'fit_metrics.json',assess([rows[i] for i in fit],training_predictions,full_controls))
        sealed={str(f):digest(f) for f in [out/'oof_predictions.json',out/'oof_metrics.json']}
        for name in models:
            for n in ('model.npz','metadata.json'):sealed[str(out/name/n)]=digest(out/name/n)
        write_json(out/'pre_calibration_seal.json',dict(model_and_OOF_sha256=sealed,calibration_predictions_started=False))
        # Independent calibration is opened exactly once, after checkpoint seal.
        for i in cal:packets[i]=validate_packet({k:v[i:i+1] for k,v in z.items()})
        cp={name:prediction_rows(model,packets,cal,rows) for name,model in models.items()}
        cc={'always_base':{i:False for i in cal},'always_F':{i:True for i in cal}}
        for cost in (0,2):cc['task_constant_cost'+str(cost)]={i:task_constant([rows[j] for j in fit],rows[i]['task'],cost) for i in cal}
        cm=assess([rows[i] for i in cal],cp,cc);verify(sealed)
        write_json(out/'calibration_predictions.json',cp);write_json(out/'calibration_metrics.json',cm)
        decision=progression(oof,cm)
        write_json(out/'result.json',dict(status='complete',fit_rows=len(fit),fit_scenes=len(groups),calibration_rows=len(cal),calibration_scenes=len({rows[i]['group'] for i in cal}),
            models_trained=2,fold_models_trained=sum(len(f['models']) for f in folds),folds_unestimable=[f['group'] for f in folds if not f['support']['coverage_passed']],
            recipe_unchanged=True,calibration_model_refits=0,seal_verified=True,OOF=oof,calibration=cm,
            progression=decision,models_calibrated=False,closed_loop_started=False,GPU_used=False,
            statistical_scope='Offline paired outcome selection on TRAIN pools; not new online success rates',
            source_sha256={str(protocol):digest(protocol),str(Path(__file__)):digest(Path(__file__)),**sealed}))
        replace_json(out/'progress.json',dict(stage='finished',progression=decision));print('RESULT',decision,flush=True)
    except BaseException:
        import traceback
        write_json(out/'failure.json',dict(traceback=traceback.format_exc(),training_may_have_started=True));raise
    return out


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--protocol',required=True,type=Path);a=ap.parse_args()
    torch.set_num_threads(2);execute(output_path(a.protocol))


if __name__=='__main__':main()
