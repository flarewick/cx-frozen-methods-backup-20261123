"""Freeze one bounded expansion after reviewed confirmation; no policy outcomes."""
import argparse
import ast
import copy
import os
from pathlib import Path

from .paths import ROOT, new_run, write_json, output_path
from .native_history_cache import read, require, verify
from .audit import digest
from .shared_onset_contract import validate_protocol as previous_protocol
from .shared_expansion_contract import validate_protocol, SEEDS


def isolation():
    sources, collisions, errors, hashes = {}, [], [], set()
    old = read(ROOT/'outputs/reports/protocols/shared_onset_interface_20261007_095007_960278/next_scene_isolation.json')
    hashes.update(old['existing_initial_hashes'])
    files = {Path(p) for p in old['source_sha256'] if Path(p).is_file()}
    names = {'manifest.json', 'episodes.json', 'records.json', 'offline_index.json', 'registry.json'}
    for base, dirs, entries in os.walk(ROOT/'outputs', followlinks=False):
        folder = Path(base)
        dirs[:] = [d for d in dirs if d not in ('robotwin','cache','source_snapshot','evaluation_source_snapshot',
            'maintenance','traces','legal_cache','__pycache__') and not (folder/d).is_symlink()]
        for name in entries:
            if name in names and not (folder/name).is_symlink(): files.add(folder/name)
    def visit(x, path, key=''):
        if isinstance(x, dict):
            for k,v in x.items():
                if k in ('initial_physical_sha256','initial_native_sha256') and isinstance(v,str): hashes.add(v)
                visit(v,path,k)
        elif isinstance(x, list):
            for value in x: visit(value,path,key)
        elif ('seed' in key.lower()) and isinstance(x,int) and not isinstance(x,bool) and x in SEEDS:
            collisions.append(dict(path=str(path),key=key,seed=x))
    for path in sorted(files):
        try:
            visit(read(path),path); sources[str(path)]=digest(path)
        except (ValueError,OSError) as exc: errors.append(dict(path=str(path),error=str(exc)))
    result=dict(passed=not collisions and not errors,collisions=collisions,errors=errors,
        sources_checked=len(sources),source_sha256=sources,existing_initial_hashes=sorted(hashes),
        proposed_seeds=SEEDS,scope='Surviving metadata plus previously registered initial hashes; deleted history cannot be reconstructed')
    require(result['passed'],'Expansion seed collision or unreadable metadata')
    return result


def unchanged_functions(old, new, names):
    def definitions(path):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body
                if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a,b=definitions(old),definitions(new)
    for name in names: require(a[name]==b[name],'Unchanged runtime function '+name)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--parent',required=True,type=Path)
    ap.add_argument('--review',required=True,type=Path)
    ap.add_argument('--cleanup',required=True,type=Path)
    a=ap.parse_args()
    parent,review,cleanup=map(output_path,(a.parent,a.review,a.cleanup))
    original=read(parent);previous_protocol(original);verify(original['source_sha256'])
    r=read(review/'result.json');verify(r['source_sha256'])
    require(r['status']=='passed' and r['complete_rollouts']==192 and r['distinct_scenes']==24,'Reviewed confirmation')
    clean=read(cleanup/'result.json')
    require(clean['status'] in ('verified_complete','verified_partial') and clean['data_or_models_deleted']==0
        and clean['write_fsync_1MiB_passed'],'Verified scoped cleanup before expansion')
    unchanged_functions(ROOT/'cx_tta/collect_shared_onset.py',ROOT/'cx_tta/collect_shared_expansion.py',['rollout'])
    unchanged_functions(ROOT/'cx_tta/shared_onset_analysis.py',ROOT/'cx_tta/shared_expansion_analysis.py',['summarize'])
    iso=isolation()
    out=new_run('shared_onset_expansion',group=('reports','protocols'))
    write_json(out/'isolation.json',iso)
    p=copy.deepcopy(original)
    p.update(stage='shared_onset_expansion',status='frozen_before_expansion_eligibility_and_policy_outcomes',
        seeds=SEEDS,scene_split={str(s):'confirmation' for s in SEEDS},maximum_formal_rollouts=512,
        parent_protocol=str(parent),parent_result=str(review/'result.json'),
        role='One finite frozen-model two-task expansion; no refit, threshold search, replacement or repeated extension',
        isolation=str(out/'isolation.json'),isolation_sha256=digest(out/'isolation.json'),
        stop_rule='Complete fixed 32 attempts per task and both blocks; invalid scenes never replaced; no outcome-driven stopping',
        source_panel='New expansion only; report independently of the already viewed 501500-501515 confirmation panel',
        primary_comparison='Frozen G_candidate_cost2 vs contemporaneous Z_zero; retain F_first_full and G_observation_cost2',
        next_task_progression=dict(each_task_net_gain_strictly_positive=True,
            each_process_pooled_net_gain_strictly_positive=True,
            pooled_harm_less_than_unconditional_F=True,pooled_rescue_minus_two_harm_strictly_positive=True,
            interpretation='Prospective research progression screen, not statistical significance or safety guarantee'),
        reporting=['complete/censored/invalid','per-task and per-process paired rescue/harm/net',
            'base-success preservation denominator','opportunity/acceptance coverage',
            'whole-scene clustered uncertainty with both process repeats retained'],
        local_scope='Expert eligibility only; VLA formal expansion externally on GPU0',
        process_wall_time_limit_s=21600,cleanup_result=str(cleanup/'result.json'),
        uncertainty='Fixed finite robustness panel; no claim of sufficient power or calibrated risk probability')
    sources=dict(original['source_sha256'])
    paths=[parent,review/'result.json',review/'verification.json',cleanup/'result.json',out/'isolation.json',
        Path(__file__).resolve(),ROOT/'tests/test_shared_expansion_contract.py']
    paths += [ROOT/'cx_tta'/f'{n}.py' for n in ('shared_expansion_contract','shared_expansion_analysis',
        'collect_shared_expansion','screen_shared_expansion','ready_shared_expansion','review_shared_expansion')]
    paths += [ROOT/'scripts'/f'{n}.sh' for n in ('eval_shared_expansion','screen_shared_expansion')]
    for path in paths:sources[str(path)]=digest(path)
    p['source_sha256']=sources;validate_protocol(p)
    write_json(out/'protocol.json',p)
    write_json(out/'static_validation.json',dict(passed=True,rollout_AST_identical=True,numeric_audit_AST_identical=True,
        model_and_thresholds_unchanged=True,source_files_verified=len(sources),new_policy_outcomes=0))
    print('PROTOCOL',out/'protocol.json',flush=True)


if __name__=='__main__':main()
