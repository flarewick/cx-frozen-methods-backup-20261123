"""Independent read-back: scene grouping, causal inputs, native labels and RNG."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .audit import digest
from .cache_recovery import read,verify_hashes,legal_observation
from .eval_diagnostics import array_hash
from .audit_recovery_cache import restore_json_rng
from .paths import write_json


def audit(run):
    p=read(run/'protocol.json');r=read(run/'result.json');verify_hashes(p['source_sha256'])
    env=read(run/'environment.json');n=len(p['rows']);nf=len(p['feature_rows'])
    if r['status']!='complete_pending_audit' or not env['golden_exact'] or not env['compiled'] or env['global_dynamo_disabled'] or env['model_trainable_parameters']:
        raise ValueError('Frozen compiled cache incomplete')
    if digest(run/'data.npz')!=r['data_sha256'] or digest(run/'features.npz')!=r['features_sha256']:raise ValueError('Changed arrays')
    with np.load(run/'data.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(run/'features.npz') as z:f={k:z[k].copy() for k in z.files}
    if f['joint'].shape!=(nf,28672) or f['joint'].dtype!=np.float16 or f['state'].shape!=(nf,14):raise ValueError('Feature shape')
    for key in ('context','semantic','coarse'):
        if d[key].shape!=(n,157):raise ValueError('Context shape')
    for key in ('base','repeated_base','target'):
        if d[key].shape!=(n,32,14):raise ValueError('Action shape')
    if any(not np.isfinite(x).all() for x in list(d.values())+list(f.values())):raise ValueError('Nonfinite data')
    feature_checks=read(run/'feature_checks.json')
    if len(feature_checks)!=nf or len(r['rows'])!=n:raise ValueError('Missing feature/query records')
    from .dataset import context_features
    coarse=[]
    for i,row in enumerate(p['feature_rows']):
        ims,state,sig=legal_observation(row);check=feature_checks[i]
        if not np.array_equal(state,f['state'][i]) or check['signature']!=sig or not check['repeat_exact'] or not check['rng_unchanged']:
            raise ValueError('Feature provenance failed')
        coarse.append(context_features(ims,state)[14:])
    current=np.asarray(p['feature_current_indices']);hist=np.asarray(p['feature_history_indices'])
    if not np.array_equal(current,d['feature_current_indices']) or not np.array_equal(hist,d['feature_history_indices']):raise ValueError('Feature indices')
    # Independently reconstruct the preprocessing, including state/pi and external fork history.
    for name,features,prepath in [('semantic',f['joint'],p['preprocess']),('coarse',np.stack(coarse),p['coarse_preprocess'])]:
        with np.load(prepath) as pre:
            v=features.astype(np.float32)@pre['pca_components'].T
            v-=pre['pca_mean'].reshape(1,-1)@pre['pca_components'].T
            state=f['state'].copy();state[:,[0,1,2,3,4,5,7,8,9,10,11,12]]/=np.pi
            observed=np.concatenate((v,state),axis=1);expected=[]
            for i in current:
                j=hist[i]
                if not 0<=j<i:raise ValueError('Expected real past feature')
                x,y=p['feature_rows'][i],p['feature_rows'][j]
                if x['group']!=y['group'] or x['scene_group']!=y['scene_group'] or x['native_count']-y['native_count']!=32 or x['past_observation']!=y['observation']:
                    raise ValueError('Cross-branch/future feature history')
                combined=np.concatenate((observed[i],observed[j],np.array([1],np.float32)))
                expected.append((combined-pre['context_mean'])/pre['context_std'])
            if not np.allclose(d[name],np.stack(expected),rtol=1e-5,atol=2e-5):raise ValueError('Preprocessing mismatch '+name)
    if not np.array_equal(d['semantic'],d['context']):raise ValueError('Semantic alias mismatch')
    from .recovery_contract import rng_fingerprint
    previous={};checks=[];rngfiles=[];first_rng=[]
    for i,(row,q) in enumerate(zip(p['rows'],r['rows'])):
        if p['feature_rows'][current[i]]!=row:raise ValueError('Training/feature row mismatch')
        if q['observation_signature']!=row['observation_signature'] or q['group']!=row['group'] or q['native_count']!=row['native_count']:
            raise ValueError('Base query at wrong predecessor')
        source=Path(row['source']);res=read(source/'result.json');trace=[json.loads(x) for x in (source/'trace.jsonl').read_text().splitlines()]
        off=row['native_count']-res['prefix_actions'];count=min(32,len(trace)-off)
        if count!=row['valid_actions'] or not 0<count<=32 or row['native_count']+count>res['native_limit']:
            raise ValueError('Teacher budget')
        if not np.array_equal(d['valid'][i],np.arange(32)<count) or np.any(d['target'][i,count:]!=0):raise ValueError('Tail mask')
        if not np.array_equal(d['target'][i,:count],np.asarray([x['action'] for x in trace[off:off+count]],np.float32)):
            raise ValueError('Nonexecuted teacher label')
        if trace[off]['before_observation']!=row['observation_signature']:raise ValueError('Label predecessor')
        for j in range(32):
            expected_count=row['native_count']+j+1 if j<count else -1
            expected_plan=trace[off+j]['plan'] if j<count else -1
            if d['source_native_counts'][i,j]!=expected_count or d['plan_ids'][i,j]!=expected_plan:raise ValueError('Wrong native/plan alignment')
        with np.load(row['target']) as z:
            if any(not np.array_equal(z[k],d[k][i]) for k in ('target','valid','source_native_counts','plan_ids')):raise ValueError('Target file mismatch')
        j=previous.get(row['group'],-1)
        if j!=p['history_indices'][i] or j!=d['history_indices'][i]:raise ValueError('Wrong training-row history')
        if j>=0 and r['rows'][j]['after_rng']!=q['before_rng']:raise ValueError('Broken sampler stream')
        if j<0:first_rng.append(q['before_rng'])
        previous[row['group']]=i;rngfile=run/f'rng_{i:02d}.json';rngfiles.append(rngfile);rng=read(rngfile)
        for when in ('before','after'):
            if rng_fingerprint(restore_json_rng(rng[when]))!=rng[when]['sha256'] or rng[when]['sha256']!=q[when+'_rng']:raise ValueError('Full RNG provenance')
        if q['before_rng']==q['after_rng'] or q['after_rng']!=q['repeat_after_rng']:raise ValueError('Sampler noise advancement')
        exact=bool(np.array_equal(d['base'][i],d['repeated_base'][i]));delta=float(np.abs(d['base'][i]-d['repeated_base'][i]).max())
        if array_hash(d['base'][i])!=q['base_sha256'] or exact!=q['repeat_exact'] or delta!=q['repeat_max_abs']:raise ValueError('Base repeat accounting')
        checks.append(dict(row=i,scene_group=row['scene_group'],branch=row['group'],valid_actions=count,repeat_exact=exact,repeat_max_abs=delta))
    if len(set(first_rng))!=1:raise ValueError('Unequal initial branch RNG')
    scenes=sorted({x['scene_group'] for x in p['rows']});teacher_steps=sum(x['valid_actions'] for x in checks)
    caps=np.array([.1]*6+[.2]+[.1]*6+[.2],np.float32)
    residual=np.abs(d['target']-d['base']);valid=d['valid'];within=np.all(residual<=caps+1e-7,axis=-1)
    cap_report=dict(arm_rad=.1,gripper_normalized=.2,valid_commands=int(valid.sum()),
        exact_teacher_commands_within_cap=int((within & valid).sum()),
        fraction_within_cap=float((within & valid).sum()/valid.sum()),
        max_required_arm_residual=float(residual[:,:, [0,1,2,3,4,5,7,8,9,10,11,12]][valid].max()),
        scope='Saved same-state command-distance diagnostic only, not closed-loop reachability or success under clipping')
    report=dict(passed=True,schema='fixed_recovery_context_v1_state_div_pi_real_fork_history',training_ready_rows=n,
        independent_scenes=scenes,branches=len(previous),valid_teacher_actions=teacher_steps,
        repeat_exact_rows=sum(x['repeat_exact'] for x in checks),max_repeat_difference=max(x['repeat_max_abs'] for x in checks),
        checks=checks,original_cap_diagnostic=cap_report,source_sha256={str(f):digest(f) for f in [run/'protocol.json',run/'result.json',run/'data.npz',run/'features.npz',run/'inventory.json',run/'environment.json',run/'feature_checks.json',Path(__file__),*rngfiles]},
        approved_scope='Developer same-task whole-scene holdout preparation only; branch forks are correlated. No training launched or adapter efficacy demonstrated.')
    write_json(run/'audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('checks','source_sha256')},indent=2))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);a=p.parse_args();threadpool_limits(2);audit(a.run.resolve())

if __name__=='__main__':main()
