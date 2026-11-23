"""Independent identity/budget/input/base-pair audit for a label inventory."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash
from .paths import write_json

def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();result=read(run/'result.json');checks=[];sources=[run/'result.json',Path(__file__)]
    if not result['passed'] or result['status']!='complete':raise ValueError('Completed inventory required')
    for file,sha in result['source_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Inventory source changed '+file)
    for r in result['records']:
        source=Path(r['source']);manifest=read(source/'manifest.json');native=read(source/'result.json');trace=[json.loads(x) for x in (source/'trace.jsonl').read_text().splitlines()];offset=r['native_count']-native['prefix_actions'];count=min(32,len(trace)-offset)
        if r['instruction']!=manifest['instruction'] or r['scene_seed']!=manifest['scene_seed'] or r['native_limit']!=native['native_limit']:raise ValueError('Teacher identity/budget mismatch')
        if offset<0 or offset%32 or count!=r['valid_actions']:raise ValueError('Decision scheduling mismatch')
        expected=np.zeros((32,14),np.float32);expected[:count]=np.array([x['action'] for x in trace[offset:offset+count]],np.float32)
        target=Path(r['target']);sources.append(target)
        if digest(target)!=r['target_sha256']:raise ValueError('Target changed')
        with np.load(target) as z:
            if not np.array_equal(z['target'],expected) or not np.array_equal(z['valid'],np.arange(32)<count) or not np.array_equal(z['source_native_counts'][:count],[x['action_count'] for x in trace[offset:offset+count]]):raise ValueError('Actual execution labels differ')
        observation=Path(r['observation'])
        if digest(observation)!=r['observation_sha256']:raise ValueError('Observation changed')
        with np.load(observation) as z:
            sig=dict(images={cam:array_hash(z[prefix+'_rgb']) for cam,prefix in [('head_camera','head'),('right_camera','right'),('left_camera','left')]},command_state=array_hash(z['command_state_raw']))
        expected_signature=trace[offset]['before_observation']
        if sig!=r['observation_signature'] or sig!=expected_signature:raise ValueError('Image/command is not label predecessor')
        expected_past=None if offset==0 else source/('fork.npz' if offset==32 else f'observation_{offset-32:04d}.npz')
        if r['past_observation']!=(None if expected_past is None else str(expected_past)):raise ValueError('History is not causal')
        if r['training_approved'] or not r['missing']:raise ValueError('Incomplete row incorrectly approved')
        for match in r['matched_compiled_base']:
            basefile=Path(match['file']);parent=basefile.parent.parent;bm=read(parent/'manifest.json');protocol_path=Path(bm['protocol']);sources.extend([parent/'manifest.json',protocol_path]);protocol=read(protocol_path)
            if digest(protocol_path)!=bm['protocol_sha256']:raise ValueError('Compiled protocol changed')
            point=protocol['points'][bm['point']]
            if point['instruction']!=r['instruction'] or bm['scene_seed']!=r['scene_seed'] or bm['source']!=manifest['source']:raise ValueError('Compiled instruction/checkpoint source mismatch')
            provenance=Path(bm['source'])/'provenance.json';sources.append(provenance)
            with np.load(basefile) as z:base=z['actions']
            if base.shape!=(32,14) or not np.isfinite(base).all() or digest(basefile)!=match['sha256']:raise ValueError('Invalid matched base')
            query=next(x for x in [json.loads(s) for s in (basefile.parent/'queries.jsonl').read_text().splitlines()] if x['query']==match['query'])
            physical=read(source/'fork.json')['diagnostics']['physical'] if offset==0 else trace[offset-1]['offline_only']['physical']
            if query['observation']!=sig or query['physical']!=physical or query['native_count']!=r['native_count'] or array_hash(base)!=query['base_sha256']:raise ValueError('Same-state compiled pair mismatch')
        checks.append(dict(group=r['group'],native_count=r['native_count'],verified_actions=count,role=r['role'],same_state_base=bool(r['matched_compiled_base'])))
    write_json(run/'audit.json',dict(passed=True,rows=len(checks),checks=checks,source_sha256={str(f):digest(f) for f in sorted(set(sources))},scope='Actual executed target/input identity and causal/budget/source audit only; no automatic semantic approval, trained model, new recovery success or dataset promotion.'))
    print(json.dumps(dict(passed=True,rows=len(checks),verified_target_actions=sum(x['verified_actions'] for x in checks),training_ready=0),indent=2))

if __name__=='__main__':main()
