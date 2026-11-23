"""Freeze new-scene gate confirmation only after both old-scene smokes pass."""
import argparse
from pathlib import Path
from .paths import ROOT,new_run,write_json,output_path
from .native_history_cache import read,require,verify
from .audit import digest
from .shared_onset_contract import CONDITIONS,TASKS,SEEDS,validate_protocol


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--interface-protocol',type=Path,required=True);ap.add_argument('--smokes',type=Path,nargs=2,required=True)
    a=ap.parse_args();parent=output_path(a.interface_protocol);p=read(parent);validate_protocol(p);verify(p['source_sha256'])
    isolation=parent.parent/'next_scene_isolation.json';iso=read(isolation)
    require(iso['passed'] and not iso['collisions'] and iso['proposed_seeds']==SEEDS,'Prospective scene isolation')
    checks={};sources=dict(p['source_sha256'])
    for run in a.smokes:
        run=output_path(run);m=read(run/'manifest.json');r=read(run/'result.json')
        require(m['mode']=='smoke' and m['protocol_sha256']==digest(parent) and m['conditions']==list(CONDITIONS),'Exact interface protocol')
        require(r['interface_passed'] and r['complete_rollouts']==0 and r['censored']==4 and len(r['branch_equivalence'])==2 and all(v['passed'] for v in r['branch_equivalence']),'Bounded actual selected-branch identity')
        require(read(run/'storage_cleanup.json')['removed'] and read(run/'source_runtime_audit.json')['passed'],'Cleaned immutable runtime')
        require(read(run/'independent_branch_check.json')['passed'],'Independent branch check')
        verify(r['source_sha256']);require(m['task'] not in checks,'One smoke per task');checks[m['task']]=str(run)
        for n in ('manifest.json','result.json','storage_cleanup.json','source_runtime_audit.json','independent_branch_check.json'):sources[str(run/n)]=digest(run/n)
    require(set(checks)==set(TASKS),'Both tasks validated')
    p.update(stage='shared_onset_confirmation',status='frozen_before_new_scene_eligibility_and_gate_outcomes',seeds=SEEDS,
        scene_split={str(s):'confirmation' for s in SEEDS},confirmation_only=True,maximum_formal_rollouts=256,
        role='New whole-scene confirmation of frozen empirical gate; never fit or retune on these scenes',
        labels='Paired complete Z/F/G outcomes for confirmation only; all censored episodes retained',
        parent_protocol=str(parent),parent_result=str(Path(p['gate_run'])/'result.json'),interface_smokes=checks,
        isolation=str(isolation),isolation_sha256=digest(isolation),no_new_training_or_TTA=True,
        stop_rule='Complete fixed eligible scenes and both blocks; no outcome-based stopping or replacement',
        primary_comparison='Frozen G_candidate_cost2 vs Z_zero; F and G_observation_cost2 remain fixed controls',
        success_criteria='Report each task and independent process: net rescue-minus-harm, base-success preservation, selection coverage and paired scene outcomes. No post-hoc winner/threshold selection.',
        preservation='All fit/calibration/old confirmation scenes excluded. This panel cannot enter refitting or threshold selection.',
        candidate_scope='One original ARM-only F chunk or exact base; rejection consumes sole opportunity; not new-task transfer',
        probability_scope='Uncalibrated scores; no safety probability claim')
    for k in ('registry_source_protocol','supersedes','future_training','insufficient_support','justification'):p.pop(k,None)
    files=[parent,isolation,Path(__file__).resolve(),ROOT/'cx_tta/screen_shared_onset.py',ROOT/'scripts/screen_shared_onset.sh',ROOT/'cx_tta/ready_shared_onset.py',ROOT/'cx_tta/check_shared_onset_branches.py',ROOT/'tests/test_empirical_onset_gate.py']
    for f in files:sources[str(f)]=digest(f)
    p['source_sha256']=sources;validate_protocol(p)
    out=new_run('shared_onset_confirmation',group=('reports','protocols'));write_json(out/'protocol.json',p)
    print('PROTOCOL',out/'protocol.json',flush=True)

if __name__=='__main__':main()
