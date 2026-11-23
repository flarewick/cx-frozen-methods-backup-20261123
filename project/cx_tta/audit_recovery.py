"""Independently validate terminal-continuation artifacts before expansion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .audit_correction_budget import CAPS
from .eval_diagnostics import array_hash,json_hash
from .paths import ROOT,write_json
from .recovery_contract import terminal_label,comparison


def saved_signature(file):
    with np.load(file) as z:
        return dict(images={camera:array_hash(z[name+'_rgb']) for name,camera in [('head','head_camera'),('right','right_camera'),('left','left_camera')]},command_state=array_hash(z['command_state_raw']))


def audit(run):
    manifest=json.loads((run/'manifest.json').read_text());result=json.loads((run/'result.json').read_text())
    if result['status']!='complete' or not result['gate_passed']:raise ValueError('Run is censored/incomplete or gate failed')
    if digest(Path(manifest['protocol']))!=manifest['protocol_sha256']:raise ValueError('Changed protocol')
    for path,sha in manifest['code_hashes'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Changed runner dependency '+path)
    if manifest.get('compile_cache_source'):
        copied=json.loads((run/'compile_cache_copy.json').read_text());origin=Path(copied['source'])
        if str(origin)!=manifest['compile_cache_source'] or digest(origin/'manifest.json')!=copied['source_manifest_sha256']:
            raise ValueError('Compiler cache source mismatch')
        if not copied['files'] or any(digest(origin/rel)!=sha for rel,sha in copied['files'].items()):
            raise ValueError('Changed compiler cache source')
    from .cache_expert import CHECKPOINT
    from .evaluate_b2 import resolve_tokenizer
    provenance=json.loads((Path(manifest['point']['source'])/'provenance.json').read_text())
    hashes={name:digest(path) for name,path in {
        'base_checkpoint_sha256':CHECKPOINT/'model.safetensors',
        'normalization_sha256':CHECKPOINT/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json',
        'tokenizer_sha256':resolve_tokenizer()}.items()}
    if any(value!=provenance[name] for name,value in hashes.items()):raise ValueError('Base model/normalization/tokenizer differs from source')
    checked=[];sham_checks=[];lookup={};file_hashes={}
    for row in result['reports']:
        key=(row['policy_seed'],row['branch'])
        if key in lookup:raise ValueError('Duplicate branch')
        lookup[key]=row
        folder=run/f'policy_seed_{row["policy_seed"]}'/row['branch'];trace=folder/'trace.jsonl'
        if digest(trace)!=row['trace_sha256']:raise ValueError('Changed trace')
        records=[json.loads(line) for line in trace.read_text().splitlines()]
        with np.load(folder.parent/'proposal.npz') as z:proposal=dict(z)
        if np.any(np.abs(proposal['delta'])>CAPS+1e-6):raise ValueError('Proposal cap violation')
        direction=0 if row['branch'] in ('base','sham') else (1 if row['branch']=='plus' else -1)
        expected_first=proposal['base']+direction*proposal['delta']
        count=manifest['prefix_actions'];h=hashlib.sha256();observations=0
        for chunk,record in enumerate(records):
            if record['chunk']!=chunk or len(record['per_action'])!=32 or record['observation_updates']!=32:raise ValueError('Nonofficial observation cadence')
            actions=np.asarray(record['actions'],np.float32)
            if chunk==0:
                if not np.array_equal(actions,expected_first) or record['base_sha256']!=array_hash(proposal['base']):raise ValueError('Wrong intervention chunk')
            elif record['base_sha256']!=array_hash(actions):raise ValueError('Residual continued beyond first chunk')
            advanced=0
            for i,step in enumerate(record['per_action']):
                if step['attempt']!=i or step['advanced'] not in (0,1):raise ValueError('Invalid action attempt')
                if step['advanced']==0 and count<manifest['native_limit'] and not step['success']:raise ValueError('Unexplained no-op')
                count+=step['advanced'];advanced+=step['advanced'];observations+=1
                if step['count']!=count or count>manifest['native_limit']:raise ValueError('Count/budget mismatch')
            if record['executed']!=advanced:raise ValueError('Wrong executed count')
            file=folder/f'observation_{chunk:04d}.npz'
            signature=saved_signature(file)
            if signature!=record['observation']:raise ValueError('Decision observation mismatch')
            file_hashes[str(file.relative_to(run))]=digest(file);h.update(json_hash(record).encode())
        if h.hexdigest()!=row['trajectory_sha256'] or count!=row['action_count'] or observations!=row['observation_updates'] or len(records)!=row['chunks']:
            raise ValueError('Branch aggregate mismatch')
        if row['native_limit']!=manifest['native_limit'] or row['prefix_actions']!=manifest['prefix_actions'] or count-row['prefix_actions']!=row['continuation_actions']:
            raise ValueError('Budget reset or prefix mismatch')
        expected=terminal_label(count,manifest['native_limit'],records[-1]['per_action'][-1]['success'])
        if any(row[k]!=expected[k] for k in ('complete','success','reason')):raise ValueError('Incorrect terminal label')
        rng=json.loads((folder.parent/'rng.json').read_text())
        if row['start_rng']!=rng['after_common_query']['sha256']:raise ValueError('Unequal fork RNG')
        if saved_signature(folder.parent/'fork.npz')!=manifest['point']['observation_signature']:raise ValueError('Wrong recorded fork observation')
        if saved_signature(folder/'terminal.npz')!=records[-1]['per_action'][-1]['observation']:raise ValueError('Wrong terminal observation')
        interface=json.loads((folder.parent/'interface.json').read_text())
        if not interface['compiled_base_exact'] or not interface['rng_exact'] or not interface['physical']['matched']:raise ValueError('Interface gate failed')
        checked.append(dict(policy_seed=row['policy_seed'],branch=row['branch'],success=row['success'],actions=count))
    if len(lookup)!=manifest['continuations']:raise ValueError('Missing branch')
    pairs=[]
    for n,seed in enumerate(manifest['policy_seeds']):
        base=lookup[(seed,'base')]
        required=manifest['phase']=='gate' or n==0
        if required:
            sham=lookup[(seed,'sham')]
            if any(base[k]!=sham[k] for k in ('trajectory_sha256','success','action_count','start_rng')):raise ValueError('Base/sham mismatch')
            sham_checks.append(dict(policy_seed=seed,exact=True))
        for name in ('plus','minus'):pairs.append(dict(policy_seed=seed,branch=name,comparison=comparison(base,lookup[seed,name],True)))
    if pairs!=result['pairs']:raise ValueError('Paired conclusion mismatch')
    return dict(status='passed',phase=manifest['phase'],branches=checked,sham_checks=sham_checks,pairs=pairs,
        saved_observation_sha256=file_hashes,base_asset_hashes=hashes,result_sha256=digest(run/'result.json'),manifest_sha256=digest(run/'manifest.json'),
        scope='Artifact/interface validation only, not independent scene success-rate gain or reliable preference training labels.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    result=audit(args.run.resolve());write_json(args.run/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='saved_observation_sha256'},indent=2))


if __name__=='__main__':main()
