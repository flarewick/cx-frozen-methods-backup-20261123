"""S1x frozen component and consecutive-block selection diagnostics."""
import argparse
import time
import traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .paths import ROOT,new_run,write_json,replace_json
from .audit import digest
from .probe_candidates import read,verify,load_candidate
from .candidate_diagnostic import fixed_actions,row_metrics,summarize
from .component_selector import RULES,STATES,FLOORS,component_grid,choose,assemble,benefit_flags
from .recovery_fit_metrics import endpoints
from .robot_kinematics import RobotFK

SOURCE=ROOT/'outputs/reports/analyses/candidates_20261002_095759_842509'


def prepare(source):
    prior=read(source/'protocol.json');verify(prior)
    if not read(source/'audit.json')['passed']:raise ValueError('Audited S1w source required')
    result=read(source/'result.json');coverage=read(Path(prior['source'])/'protocol.json')
    pairs=coverage['seam_pairs'];chains=[]
    with np.load(prior['data']) as z:valid=z['valid'].copy()
    for group in sorted({e['group'] for e in pairs}):
        edge=sorted([e for e in pairs if e['group']==group],key=lambda e:e['before']);ids=[edge[0]['before']]+[e['after'] for e in edge]
        for j,e in enumerate(edge):
            if e['before']!=ids[j] or e['last']!=int(valid[e['before']].sum())-1 or prior['rows'][e['before']]['frame']+e['last']+1!=prior['rows'][e['after']]['frame']:
                raise ValueError('Unverified/nonconsecutive chain')
            if prior['rows'][e['before']]['kind']!='recovery' or not e['same_plan']:raise ValueError('Only audited teacher recovery links allowed')
        chains.append(dict(group=group,rows=ids,pairs=edge))
    if len(pairs)!=4 or sorted(i for c in chains for i in c['rows'])!=prior['parts']['recovery']:
        raise ValueError('Unexpected recovery continuity support')
    files=[source/f for f in ('protocol.json','result.json','audit.json')]+[Path(prior['source'])/'protocol.json']
    files += [Path(x['panel']) for x in result['reports']]
    files += [ROOT/'cx_tta'/f for f in ('component_selector.py','probe_component_selection.py','audit_component_selection.py',
        'summarize_component_selection.py','observable_candidate_scores.py','probe_observable_scores.py',
        'train_semantic_pilot.py','recovery_context_v2.py')]
    preprocess=Path(coverage['expert_source'])/'semantic_preprocess.npz'
    expert_protocol=Path(coverage['expert_source'])/'protocol.json'
    context_audit=Path(coverage['context_source'])/'audit.json'
    context_protocol=Path(read(context_audit)['source'])/'protocol.json'
    files += [preprocess,expert_protocol,context_audit,context_protocol]
    run=new_run('component_selection',group=('reports','analyses'))
    write_json(run/'protocol.json',dict(status='frozen_before_diagnostic',reference_source=str(source),
        rows=prior['rows'],parts=prior['parts'],models=prior['models'],data=prior['data'],config=prior['config'],
        feature_preprocess=str(preprocess),expert_rows_protocol=str(expert_protocol),recovery_context_audit=str(context_audit),
        rules=RULES,states=list(STATES),alphas=prior['alphas'],chains=chains,tolerance_floors=FLOORS,
        source_sha256={**prior['source_sha256'],**{str(f):digest(f) for f in files}},
        score='Same S1w dense BC+SE3 surrogate. Finger contribution is replaced independently; FK unaffected by finger command. '
            'No within-block penalty in selection, matching S1w. Report actual within-block velocity and seams.',
        continuity='Only four audited recovery edges. For each scene/arm minimize mean block dense score + '
            'mean edge native increment-error MSE normalized by0.1 arm/0.2 grip (coefficient1, S1t scales). '
            'This preserves the original balanced recovery-row to seam-loss ratio. Exact finite-state dynamic programming; '
            'nonconsecutive expert/anchor rows remain independent. No smoothing target velocities to zero.',
        guards='Strict: S1w no component regression. Tolerant: mean angle/position/grip bounded by max(base,error_floor), '
            'floors1deg/1cm/.02 from existing preservation gate; same numerical tolerances and no increased range counts. '
            'No per-task exceptions, searching or post-result threshold adjustment. All are teacher-known diagnostics.',
        comparisons='2 supports x2 tied/split arm-finger scales x2 strict/tolerant guards x2 local/sequence rules=16, '
            'all12 models. Tied strict local must reproduce S1w guarded-arm. Each joint choice uses future teacher '
            'commands and has look-ahead; this is NOT a causal router or temporal execution proof.',
        benefit='Retain S1w >5% score plus strict component guard for comparable joint benefits; also separate arm-only '
            '(native6/7 BC+rotation+position) and finger-MSE >5% gains with component nonregression. '
            'Report same component alpha across all3 seeds and distinguish finger-only from arm correction.',
        observable_probe=dict(scores=['low_disagreement','relative_agreement','low_correction_magnitude','near_fit_context'],
            scope='Nontrained observable confidence proxies only, all4 conditions. Inputs frozen157-d causal context, '
                'base and3 fixed candidate action chunks, all32 predicted steps. NO teacher, valid-tail, success, '
                'frame/phase or task-ID input. Nearest-context reference uses only856 expert-fit rows.',
            labels='Strict component benefit at a common positive alpha across3 seeds; raw and capped supports separately. '
                'Expert inner-held220 rows/12 scenes primary; fit and recovery not used to train/calibrate a probe. '
                'Successful held anchors are negatives by base target identity, not a positive discrimination test.',
            metrics='Scene/task-balanced ROC-AUC with per-task AUC; top10% rank by observable score only (including ties), '
                'report actual weighted coverage and precision. This is an evaluation ranking, not a calibrated deployment threshold. '
                'Task-stratified scene bootstrap1000 draws, fixed seed621047; percentile intervals descriptive only. '
                'Random-ranking precision baseline=weighted label prevalence. No score or sign selection after results.',
            limitations='Three scenes per task, reused developer data, and no independent recovery positives. Proxy failure '
                'does not prove all observable representations cannot identify benefit. Learned probes/router still require '
                'scene-wise candidate cross-fitting on fit data; current developer labels cannot be fed back into training.'),
        scope='S1x offline teacher-only component/sequence diagnosis then fixed observable-proxy check. Existing models, '
            'labels, preprocessing and source predictions frozen. No model/GPU inference, model/router fitting, '
            'new collection, simulation, TTA or task-success claim.'))
    print(run,flush=True)


def execute(run):
    p=read(run/'protocol.json');verify(p);start=time.monotonic();reports=[]
    if (run/'result.json').exists():raise ValueError('Fresh run required')
    reference=read(Path(p['reference_source'])/'result.json')
    with np.load(p['data']) as z:arrays={k:z[k].copy() for k in z.files}
    base=arrays['base'].astype(float);target=arrays['target'].astype(float);valid=arrays['valid']
    robots=[RobotFK(read(Path(p['config'])),s) for s in ('left','right')];tp=endpoints(target,robots)
    baseline=row_metrics(base,target,valid,robots,tp);parity=0.
    try:
        for j,(model,old) in enumerate(zip(p['models'],reference['reports'])):
            if (model['condition'],model['seed'])!=(old['condition'],old['seed']):raise ValueError('Model ordering differs')
            original=load_candidate(model,p,arrays);fixed=fixed_actions(base,original)
            with np.load(old['panel']) as z:
                oldpanel={k:z[k].copy() for k in z.files};fm=[{k:v[i] for k,v in oldpanel.items() if k!='choices'} for i in range(7)]
            grids={s:component_grid(fixed,fm,target,valid,s) for s in ('raw','capped')}
            panels={};summaries={};objectives={};all_metrics=[];all_choices=[]
            for support,grid in grids.items():
                panels[support+'_grid_score']=grid['score']
                for kind,flags in benefit_flags(grid).items():panels[support+'_'+kind+'_benefit']=flags
            for name,setting in p['rules'].items():
                selected,records=choose(grids[setting['support']],setting,fixed,target,p['chains'])
                action=assemble(fixed,setting['support'],selected);m=row_metrics(action,target,valid,robots,tp)
                all_metrics.append(m);all_choices.append(selected);objectives[name]=records
                summaries[name]={part:summarize(m,action,target,valid,p['rows'],ids,baseline) for part,ids in p['parts'].items()}
                if setting['coupling']=='tied' and setting['guard']=='strict' and setting['timing']=='local':
                    oi=read(Path(p['reference_source'])/'protocol.json')['variants'].index(setting['support']+'_guarded_arm')
                    for key in m:
                        error=float(np.abs(m[key]-oldpanel[key][oi]).max());parity=max(parity,error)
                        if error>1e-8:raise ValueError('Tied strict local no longer matches S1w '+key)
            panels.update({k:np.stack([m[k] for m in all_metrics]) for k in baseline})
            panels['choices']=np.stack(all_choices)
            file=run/f"{model['condition']}_s{model['seed']}.npz";np.savez_compressed(file,**panels)
            reports.append(dict(condition=model['condition'],seed=model['seed'],panel=str(file),panel_sha256=digest(file),
                summaries=summaries,chain_objectives=objectives))
            replace_json(run/'progress.json',dict(completed_models=j+1,total_models=12,elapsed_seconds=time.monotonic()-start))
            print('COMPLETE_MODEL',model['condition'],model['seed'],flush=True)
        verify(p)
        write_json(run/'result.json',dict(status='complete',reports=reports,elapsed_seconds=time.monotonic()-start,
            s1w_reproduction_max_error=parity,scope=p['scope']))
    except BaseException:
        write_json(run/'failure.json',dict(traceback=traceback.format_exc(),completed_models=len(reports)));raise


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--run',type=Path)
    args=parser.parse_args();threadpool_limits(2)
    if args.prepare_only:prepare(args.source.resolve())
    elif args.run is None:parser.error('--run required')
    else:execute(args.run.resolve())

if __name__=='__main__':main()
