"""Verify new-scene registry and write four explicit external evaluation commands."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import shlex
from .paths import ROOT,output_path,write_json
from .native_history_cache import read,require,verify
from .audit import digest
from .shared_onset_contract import validate_protocol,registry_records,CONDITIONS
from .collect_shared_onset import inputs


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--registry',type=Path,required=True);a=ap.parse_args()
    a.protocol=output_path(a.protocol);a.registry=output_path(a.registry);p=read(a.protocol);validate_protocol(p);verify(p['source_sha256'])
    require(p['stage']=='shared_onset_confirmation','Formal protocol only')
    rows=registry_records(a.protocol,a.registry);iso=read(p['isolation']);old=set(iso['existing_initial_hashes'])
    hashes=[r['initial_physical_sha256'] for r in rows if r['eligible']]
    require(len(hashes)==len(set(hashes)) and not set(hashes)&old,'New distinct initial physical states')
    require(read(a.registry/'storage_cleanup.json')['removed'] and read(a.registry/'source_runtime_audit.json')['passed'],'Eligibility source/cleanup passed')
    cmds=[];checks={}
    for task in p['tasks']:
        for block in p['process_blocks']:
            opts=SimpleNamespace(protocol=a.protocol,registry=a.registry,task=task,block=block,mode='full',gpu='0',dry_run=True)
            _,_,seeds=inputs(opts);name=('cup' if task=='place_empty_cup' else 'hammer')+'_block'+str(block)
            command='OPENBLAS_NUM_THREADS=1 bash '+shlex.quote(str(ROOT/'scripts/eval_shared_onset.sh'))+' --protocol '+shlex.quote(str(a.protocol))+' --registry '+shlex.quote(str(a.registry))+' --task '+task+' --mode full --block '+str(block)+' --gpu 0'
            (a.protocol.parent/(name+'.sh')).write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+command+'\n')
            cmds.append('### '+name+'\n\n```bash\n'+command+'\n```\n')
            checks[name]=dict(task=task,block=block,seeds=seeds,records=len(seeds)*len(CONDITIONS))
    (a.protocol.parent/'evaluation_commands.md').write_text('# Frozen shared gate confirmation\n\nFour independent processes; each contains paired Z/F/G_candidate/G_observation. GPU defaults to 0. No videos/RGB. Invalid scenes are not replaced; no refit, threshold changes or outcome-based stopping.\n\n'+'\n'.join(cmds))
    result=dict(status='ready_for_external_shared_gate_confirmation',protocol=str(a.protocol),registry=str(a.registry),CLI_inputs_checked=checks,
        source_files_verified=len(p['source_sha256']),initial_state_isolation=True,eligible_scenes=len(hashes),maximum_scheduled_rollouts=sum(x['records'] for x in checks.values()),
        interface_smokes=p['interface_smokes'],no_probability_guarantee=True,formal_evaluation_started=False,
        source_sha256={str(a.protocol):digest(a.protocol),str(a.registry/'records.json'):digest(a.registry/'records.json')})
    write_json(a.protocol.parent/'readiness.json',result);print('READY',result,flush=True)

if __name__=='__main__':main()
