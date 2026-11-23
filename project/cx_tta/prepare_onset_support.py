"""One finite independent TRAIN supplement, frozen before eligibility/outcomes."""
import argparse
import copy
import os
from pathlib import Path
import numpy as np
from .paths import ROOT, new_run, write_json
from .audit import digest
from .native_history_cache import read, require, verify, arrays
from .causal_arm_protocol import identifiers
from .onset_support_contract import SEEDS, TASKS, STAGE, validate_protocol
from .onset_router_features import feature_groups

PARENT=ROOT/'outputs/reports/attribution/onset_TRAIN_review_20261007_044121_638236'
PRIOR=ROOT/'outputs/reports/protocols/onset_TRAIN_20261007_021608_309810/protocol.json'


def isolation():
    names={'manifest.json','protocol.json','training_protocol.json','records.json','episodes.json',
           'prepared.json','preparation.json','offline_index.json'}
    known=set();sources={}
    for folder,dirs,files in os.walk(ROOT/'outputs',followlinks=False):
        dirs[:]=[d for d in dirs if d not in {'robotwin','runtime','cache','source_snapshot','evaluation_source_snapshot'}
                 and not Path(folder,d).is_symlink()]
        for name in sorted(set(files)&names):
            file=Path(folder,name);value=read(file);known.update(identifiers(value));sources[str(file)]=digest(file)
            def ranges(x):
                if isinstance(x,dict):
                    start,count=x.get('seed_start'),x.get('episodes')
                    if type(start)is int and type(count)is int:known.update(('*',n) for n in range(start,start+count))
                    for v in x.values():ranges(v)
                elif isinstance(x,list):
                    for v in x:ranges(v)
            ranges(value)
    collisions=[(t,s) for t in TASKS for s in SEEDS if (t,s) in known or ('*',s) in known]
    require(not collisions,'New fixed scene identifiers already allocated: '+str(collisions))
    return dict(passed=True,collisions=collisions,sources_checked=len(sources),source_sha256=sources,
        scope='All surviving declared scene metadata; unavailable deleted history cannot be reconstructed')


def coverage():
    m=read(PARENT/'decisions/manifest.json');cache=PARENT/'decisions/observables.npz'
    require(digest(cache)==m['observables_sha256'],'Accepted legal packets unchanged')
    index=PARENT/'decisions/offline_index.json'
    require(digest(index)==m['offline_index_sha256'],'Accepted split unchanged')
    x=arrays(cache);rows=read(index)['rows'];result={}
    # Fit only; descriptive coverage is not a classifier or a benefit score.
    for task in TASKS:
        rr=[r for r in rows if r['split']=='fit' and r['task']==task]
        fs=[feature_groups({k:v[r['index']:r['index']+1] for k,v in x.items()}) for r in rr]
        result[task]=dict(rows=len(rr),scenes=len({r['group'] for r in rr}),groups={})
        for k in fs[0]:
            values=np.stack([f[k] for f in fs]);norm=np.linalg.norm(values,axis=1)
            result[task]['groups'][k]=dict(dim=values.shape[1],varying_coordinates=int(np.count_nonzero(values.std(0)>1e-6)),
                norm_min=float(norm.min()),norm_max=float(norm.max()))
    return dict(tasks=result,fit_only=True,selection_on_feature_or_outcome=False,
        conclusion='Few distinct scenes and one harm scene do not establish separability; no fit-only strata reliably identify harm. Use a finite independent panel and retain all eligible outcomes.',
        source_sha256={str(cache):digest(cache),str(index):digest(index)})


def prepare():
    evidence=read(PARENT/'decision_support.json');verify(evidence['source_sha256'])
    require(evidence['coverage_passed'] is False and evidence['missing']==[
        'fit negative: 1 distinct scenes < 2','calibration negative: no distinct scene'], 'Known independent harm support gap')
    old=read(PRIOR);verify(old['source_sha256']);isolated=isolation();cov=coverage()
    out=new_run('onset_support',group=('reports','protocols'))
    write_json(out/'scene_isolation.json',isolated);write_json(out/'fit_feature_coverage.json',cov)
    p=copy.deepcopy(old)
    p.update(stage=STAGE,seeds=SEEDS,scene_split={str(s):'fit' if i<8 else 'calibration' for i,s in enumerate(SEEDS)},
        status='frozen_before_eligibility_and_candidate_outcomes',parent_result=str(PARENT/'decision_support.json'),
        fixed_single_extension=True,outcome_based_stopping=False,maximum_formal_rollouts=128,
        role='One finite independent TRAIN support extension; not a gate evaluation or a cross-task transfer claim',
        no_automatic_next_extension=True,
        justification='Prior single harm scene cannot support scene-disjoint risk validation. Both tasks retained to avoid task-gated hardcoding; no changes to candidate or physics.',
        stop_rule='Complete all preassigned eligible Z/F episodes and both blocks, even if label coverage is reached early. Invalid scenes are recorded, never replaced. No additional panel authorized automatically.',
        insufficient_support='After this fixed extension, reconsider representation/candidate/supervision design; do not automatically extend seed range again.',
        preservation='Old fit/calibration assignments unchanged; all old confirmation scenes excluded from fitting and threshold selection.',
        future_training='Audit merged coverage first. Freeze shared small readout recipe before fitting; compare observed-base risk against candidate-conditioned benefit and harm. No task-specific threshold; no uncalibrated confidence claim.',
        isolation=str(out/'scene_isolation.json'),isolation_sha256=digest(out/'scene_isolation.json'))
    files=[PARENT/'decision_support.json',PARENT/'decisions/manifest.json',PRIOR,out/'fit_feature_coverage.json']
    files+=list((ROOT/'cx_tta').glob('*.py'))
    files += [ROOT/n for n in ('scripts/collect_onset_support.sh','scripts/screen_onset_support.sh','tests/test_onset_support.py')]
    p['source_sha256'].update({str(f):digest(f) for f in files})
    validate_protocol(p);write_json(out/'protocol.json',p)
    print('PROTOCOL',out/'protocol.json',flush=True)


def main():argparse.ArgumentParser(__doc__).parse_args();prepare()
if __name__=='__main__':main()
