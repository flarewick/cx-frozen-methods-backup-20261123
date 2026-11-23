"""Verify saved-input/action/teacher alignment and the recovery cache contract."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import digest
from .eval_diagnostics import array_hash
from .paths import write_json

def read(p):return json.loads(p.read_text())

def restore_json_rng(value):
    import torch
    def tuples(x):return tuple(tuples(y) for y in x) if isinstance(x,list) else x
    name,keys,pos,gauss,number=value['numpy']
    return dict(python=tuples(value['python']),numpy=(name,np.array(keys,dtype=np.uint32),pos,gauss,number),
                torch_cpu=torch.tensor(value['torch_cpu'],dtype=torch.uint8),
                torch_cuda=[torch.tensor(s,dtype=torch.uint8) for s in value['torch_cuda']])

def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path)
    run=parser.parse_args().run.resolve();p=read(run/'protocol.json');r=read(run/'result.json')
    if r['status']!='complete_pending_audit' or len(p['rows'])!=6 or len(r['rows'])!=6:
        raise ValueError('Six completed rows required')
    for file,sha in p['source_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Changed source '+file)
    if not read(Path(p['inventory'])/'audit.json')['passed'] or not read(Path(p['fit_diagnostic'])/'audit.json')['passed']:
        raise ValueError('Prerequisite audit missing')
    environment=read(run/'environment.json')
    if not environment['compiled'] or environment['global_dynamo_disabled'] or not environment['golden_exact']:
        raise ValueError('Policy/feature convention failed')
    if environment['model_trainable_parameters']!=0:raise ValueError('Backbone was not frozen')
    if digest(run/'data.npz')!=r['data_sha256'] or digest(run/'features.npz')!=r['features_sha256']:
        raise ValueError('Changed cache arrays')
    with np.load(run/'data.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(run/'features.npz') as z:f={k:z[k].copy() for k in z.files}
    if f['joint'].shape!=(6,28672) or f['joint'].dtype!=np.float16 or f['state'].shape!=(6,14):
        raise ValueError('Wrong feature/state shape')
    if d['context'].shape!=(6,157) or d['base'].shape!=(6,32,14) or d['target'].shape!=(6,32,14):
        raise ValueError('Wrong training shape')
    if any(not np.isfinite(v).all() for v in list(d.values())+list(f.values())):raise ValueError('Nonfinite cache')
    feature_checks=read(run/'feature_checks.json');checks=[];previous={};first_rng=[];sourcefiles=[]
    from .recovery_contract import rng_fingerprint
    for i,(row,query) in enumerate(zip(p['rows'],r['rows'])):
        obs=Path(row['observation']);target=Path(row['target'])
        if digest(obs)!=row['observation_sha256'] or digest(target)!=row['target_sha256']:
            raise ValueError('Changed source observation/label')
        with np.load(obs) as z:
            sig=dict(images={c:array_hash(z[k+'_rgb']) for c,k in
                [('head_camera','head'),('right_camera','right'),('left_camera','left')]},
                command_state=array_hash(z['command_state_raw']))
            if not np.array_equal(z['command_state'],f['state'][i]):raise ValueError('Policy state mismatch')
        if sig!=row['observation_signature'] or sig!=query['observation_signature'] or sig!=feature_checks[i]['signature']:
            raise ValueError('Query/encoder/source input mismatch')
        if not feature_checks[i]['repeat_exact'] or not feature_checks[i]['rng_unchanged']:
            raise ValueError('Feature repeat/RNG failed')
        with np.load(target) as z:
            for k in ('target','valid','source_native_counts','plan_ids'):
                if not np.array_equal(z[k],d[k][i]):raise ValueError('Teacher target/mask mismatch')
        n=row['valid_actions'];native=row['native_count'];limit=row['native_limit']
        if not np.array_equal(d['valid'][i],np.arange(32)<n) or not 0<n<=32 or native+n>limit:
            raise ValueError('Invalid native budget/mask')
        if not np.array_equal(d['source_native_counts'][i,:n],np.arange(native+1,native+n+1)):
            raise ValueError('Wrong future-command alignment')
        if np.any(d['target'][i,n:]!=0) or np.any(d['source_native_counts'][i,n:]!=-1):
            raise ValueError('Invented terminal tail')
        source=Path(row['source']);trace=[json.loads(line) for line in (source/'trace.jsonl').read_text().splitlines()]
        result=read(source/'result.json');offset=native-result['prefix_actions'];sub=trace[offset:offset+n]
        if not np.array_equal(d['target'][i,:n],np.array([x['action'] for x in sub],np.float32)):
            raise ValueError('Not actual executed teacher commands')
        if trace[offset]['before_observation']!=sig:raise ValueError('Wrong label predecessor')
        j=previous.get(row['group'],-1)
        if d['history_indices'][i]!=j or p['history_indices'][i]!=j:
            raise ValueError('Wrong history index')
        if j>=0:
            if row['past_observation']!=p['rows'][j]['observation'] or p['rows'][j]['native_count']+32!=native:
                raise ValueError('Noncausal/cross-branch history')
            if r['rows'][j]['after_rng']!=query['before_rng']:raise ValueError('Policy RNG stream discontinuity')
        else:
            if row['past_observation'] is not None:raise ValueError('First-history contract changed')
            first_rng.append(query['before_rng'])
        previous[row['group']]=i
        rngfile=run/f'rng_{i:02d}.json';sourcefiles.append(rngfile);rng=read(rngfile)
        for when in ('before','after'):
            if rng_fingerprint(restore_json_rng(rng[when]))!=rng[when]['sha256'] or rng[when]['sha256']!=query[when+'_rng']:
                raise ValueError('Full saved RNG/fingerprint mismatch')
        if query['before_rng']==query['after_rng'] or query['after_rng']!=query['repeat_after_rng']:
            raise ValueError('Internal noise/RNG advancement failed')
        if array_hash(d['base'][i])!=query['base_sha256']:raise ValueError('Wrong compiled base')
        exact=bool(np.array_equal(d['base'][i],d['repeated_base'][i]));difference=float(np.abs(d['base'][i]-d['repeated_base'][i]).max())
        if exact!=query['repeat_exact'] or difference!=query['repeat_max_abs']:raise ValueError('Repeat report mismatch')
        for old in query['matched_old_base_comparison']:
            with np.load(old['source']['file']) as z:oldbase=z['actions']
            if old['exact']!=bool(np.array_equal(d['base'][i],oldbase)) or abs(old['max_absolute_difference']-float(np.abs(d['base'][i]-oldbase).max()))>1e-8:
                raise ValueError('Old same-state base comparison differs')
        checks.append(dict(group=row['group'],native_count=native,valid_actions=n,native_limit=limit,
            history=j,repeat_exact=exact,repeat_max_abs=difference))
    if len(set(first_rng))!=1:raise ValueError('Per-scene initial sampler seed differs')
    with np.load(p['preprocess']) as z:
        # Recompute current and past explicitly, independently of the cache helper.
        visual=(f['joint'].astype(np.float32)-z['pca_mean'])@z['pca_components'].T
        current=np.concatenate((visual,f['state']),axis=1)
        expected=[]
        for i,j in enumerate(d['history_indices']):
            combined=np.concatenate((current[i],np.zeros(78,np.float32) if j<0 else current[j],np.array([j>=0],np.float32)))
            expected.append((combined-z['context_mean'])/z['context_std'])
        if not np.allclose(d['context'],np.array(expected),rtol=1e-5,atol=2e-5):raise ValueError('Frozen PCA/history context mismatch')
    groups=sorted(previous)
    if len(groups)!=2 or sum(x['valid_actions'] for x in checks)!=144:raise ValueError('Coverage mismatch')
    write_json(run/'audit.json',dict(passed=True,training_ready_rows=6,groups=groups,valid_teacher_actions=144,
        approved_scope='developer_recovery_fit_interface_only',checks=checks,
        repeat_exact_rows=sum(x['repeat_exact'] for x in checks),
        source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'data.npz',
            run/'features.npz',run/'environment.json',run/'feature_checks.json',Path(__file__),*sourcefiles]},
        scope='Paired cache/input/label/causal-history/RNG audit, no held-out group promotion or task-success claim. '
              'Repeat numerical differences, if present, are reported and the first sample is always retained.'))
    print(json.dumps(dict(passed=True,training_ready_rows=6,groups=groups,valid_teacher_actions=144,
                         repeat_exact_rows=sum(x['repeat_exact'] for x in checks)),indent=2))

if __name__=='__main__':main()
