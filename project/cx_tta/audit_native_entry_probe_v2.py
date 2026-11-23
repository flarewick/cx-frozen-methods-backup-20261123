"""Independently verify preserved raw and corrected replay evidence on CPU."""
import argparse
import copy
import json
from pathlib import Path

from .audit import digest
from .audit_native_entry_probe import audit
from .native_history_cache import read,require
from .paths import write_json
from .probe_action_replay import compare_physical


def audit_replay(run):
    run=Path(run).resolve();m=read(run/'manifest.json');p=read(m['protocol']);point=m['point']
    require(p['version']=='S2n-native-entry-active-alias-v2','Expected amended replay contract')
    source=Path(point['source']);trace=source/'traces'/f"r0_seed{point['seed']}.jsonl"
    current=[json.loads(x) for x in trace.read_text().splitlines()]
    checks=[json.loads(x) for x in (run/'replay_checks.jsonl').read_text().splitlines()]
    episodes=read(run/'episodes.json');expected=(2*point['chunk']+1)*len(episodes)
    require(len(checks)==expected,'Incomplete native source checkpoints')
    sources={str(trace):digest(trace),str(run/'replay_checks.jsonl'):digest(run/'replay_checks.jsonl'),
        str(Path(__file__)):digest(Path(__file__)),str(Path(m['protocol'])):digest(Path(m['protocol']))}
    alias_checks=0;seen=set()
    for row in checks:
        key=row['condition'],row['repeat'],row['position'],row['chunk']
        require(key not in seen,'Duplicate replay checkpoint');seen.add(key)
        index=row['chunk'];when='after' if row['position']=='after' else 'before'
        require(row['position'] in ('before','after','fork'),'Replay checkpoint type')
        require(0<=index<=point['chunk'],'Replay checkpoint outside selected prefix')
        require(row['reference']==current[index][when],'Replay reference differs from sealed source')
        count=sum(x['executed'] for x in current[:index+(when=='after')])
        require(row['native_count']==count,'Native replay counter differs')
        raw=compare_physical(row['reference'],row['actual'],1e-5);reported=row['compared']
        require(reported['matched'],'Rejected physical replay cannot be accepted')
        alias=reported.get('diagnostic_alias_correction')
        if alias is None:
            require(raw['matched'] and reported['raw_comparison']==raw,'Unamended raw comparison mismatch')
        else:
            require(raw==reported['raw_comparison'] and raw.get('reason')=='entity_set_changed'
                and raw['missing']==['object_poses/cluttered_obj'] and not raw['extra'],
                'Amendment concealed another entity mismatch')
            require(alias['fresh_task_has_alias'] is False and alias['created_clutter']==[],
                'A live diagnostic entity was excluded')
            prior=source/'traces'/f"r0_seed{alias['prior_seed']}.jsonl"
            manifest=read(source/'manifest.json');i=manifest['seeds'].index(point['seed'])
            require(i>0 and alias['prior_seed']==manifest['seeds'][i-1],'Alias prior-scene provenance')
            previous=[json.loads(x) for x in prior.read_text().splitlines()]
            pose=row['reference']['offline_only']['object_poses']['cluttered_obj']
            require(previous[-1]['after']['offline_only']['object_poses'].get('cluttered_obj')==pose
                and all(x[w]['offline_only']['object_poses'].get('cluttered_obj')==pose
                    for x in current for w in ('before','after')),'Alias origin/constancy proof failed')
            require(alias['prior_trace_sha256']==digest(prior) and alias['source_trace_sha256']==digest(trace),
                'Alias evidence hash changed')
            cleaned=copy.deepcopy(row['reference']);del cleaned['offline_only']['object_poses']['cluttered_obj']
            compared=compare_physical(cleaned,row['actual'],1e-5)
            require(compared['matched'] and compared['errors']==reported['errors']
                and compared['max_abs_error']==reported['max_abs_error'],'Real physical mismatch hidden')
            sources[str(prior)]=digest(prior);alias_checks+=1
    # The existing independent audit also verifies applied actions, exact first
    # RGB/command/RNG/base pairing, full repeats, native budgets and certificate.
    base=audit(run)
    report=dict(passed=True,replay_checkpoints_verified=len(checks),stale_alias_checkpoints=alias_checks,
        only_proven_previous_scene_alias_excluded=True,physical_tolerance=1e-5,
        exact_RGB_and_command_contract_unchanged=True,old_action_CPU_audit_passed=base['passed'],
        decision=base['decision'],source_sha256=sources)
    write_json(run/'CPU_replay_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    audit_replay(p.parse_args().run)


if __name__=='__main__':main()
