"""Independent compact-number audit of five S2o conditions."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from .audit import digest
from .native_history_cache import read,verify,require,arrays
from .output_support_adapter import CONDITIONS as CELLS,OutputSupportAdapter,validate_native_arms
from .structured_action_adapter import FORWARD_KEYS
from .eval_diagnostics import array_hash
from .structured_attribution_analysis import _load_trace,_comparison,_probe
from .train_output_support import read_checkpoint

CONDITIONS=('L0',*CELLS)


def evaluation_order(seeds,smoke=False):
    require(len(seeds)==2 and len(set(seeds))==2,'Fixed two-scene panel')
    if smoke:return [(seeds[0],0,c) for c in CONDITIONS]
    return [(s,r,c) for i,s in enumerate(seeds) for r in range(2)
        for c in (CONDITIONS if (i+r)%2==0 else tuple(reversed(CONDITIONS)))]


class InitialPairingGuard:
    def __init__(self):self.first={};self.pairs={}
    def observe(self,seed,repeat,condition,row,base):
        key=(seed,repeat,condition);require(key not in self.first,'Duplicate first query')
        self.first[key]=(row,np.asarray(base).copy());left=self.first.get((seed,repeat,'L0'))
        if left is None:return
        for c in CONDITIONS[1:]:
            k=(seed,repeat,c);right=self.first.get(k)
            if right is None or k in self.pairs:continue
            a,ab=left;b,bb=right
            checks={f:a[f]==b[f] for f in ('observation_sha256','native_before_sha256','rng_before_query','rng_after_sampler','structured_input_sha256')}
            checks['base_bytes']=ab.tobytes()==bb.tobytes();checks['no_history']=not a['has_history'] and not b['has_history']
            require(all(checks.values()),'First inputs/physical/RNG/base differ')
            self.pairs[k]=dict(seed=seed,repeat=repeat,condition=c,checks=checks,passed=True)
    def summary(self,order):
        expected={k for k in order if k[2]!='L0'}
        return dict(passed=set(self.pairs)==expected,expected_comparisons=len(expected),comparisons=list(self.pairs.values()))


def summarize(run):
    run=Path(run);m=read(run/'manifest.json');episodes=read(run/'episodes.json')
    from .evaluate_output_support import validate_protocol
    p,*_=validate_protocol(Path(m['training_run']),m['task']);verify(p['source_sha256'])
    require(digest(Path(m['evaluation_protocol']))==m['evaluation_protocol_sha256'],'Protocol identity')
    order=evaluation_order(m['seeds'],m['max_chunks'] is not None);errors=[];traces={};guard=InitialPairingGuard()
    models={}
    for c in CELLS:
        payload=read_checkpoint(Path(m['training_run'])/(c+'_seed0.pt'),c)
        model=OutputSupportAdapter(32,payload['metadata']['geometry_mean'],c).eval().requires_grad_(False)
        model.load_state_dict(payload['model']);models[c]=model
    for ep in episodes:
        key=ep['seed'],ep['repeat'],ep['condition']
        try:
            require(key in order and key not in traces,'Episode identity')
            t=_load_trace(run,ep);rows=t['rows'];z=t['arrays'];require(len(rows)==ep['queries'] and len(rows)>0,'Query count')
            for field,values in z.items():
                shape=(len(rows),14) if field=='state' else (len(rows),32,14)
                require(values.shape==shape and values.dtype==np.float32 and np.isfinite(values).all(),'Trace array contract')
            count=0;previous=None
            for i,row in enumerate(rows):
                b=z['base'][i];d=z['applied_delta'][i];raw=z['raw_delta'][i]
                require(row['evaluated_condition']==ep['condition'],'Recorded cell')
                require(row['base_chunk_sha256']==array_hash(b) and row['raw_delta_sha256']==array_hash(raw)
                    and row['applied_delta_sha256']==array_hash(d),'Actual delta/base hashes')
                require(raw.tobytes()==d.tobytes(),'Undeclared strength intervention')
                action=b if ep['condition']=='L0' else b+d
                if ep['condition']=='L0':require(not np.any(d),'L0 not exact bypass')
                require(row['action_chunk_sha256']==array_hash(action),'Executed action formula')
                if ep['condition'].startswith(('A_','C_')):
                    require(not np.any(np.abs(d)>np.asarray(p['reference_scales'])+1e-6),'Declared bounded cell exceeds range')
                require(row['robot_domain']==validate_native_arms(action,p['robot_joint_limits']),'Domain accounting differs')
                require(row['obs_updates']==32 and 0<row['executed']<=32,'Official callback/get_obs count')
                before=row['native_before']['control'];after=row['native_after']['control']
                require(before['take_action_cnt']==count and after['take_action_cnt']-count==row['executed'],'Native clock')
                count=after['take_action_cnt'];require(count<=after['step_lim'],'Budget reset/overflow')
                require(row['rng_before_query']==row['rng_after_feature'],'Encoding drew RNG')
                require(row['base_infer_calls']==1 and row['auxiliary_prefix_calls']==1,'Extra query')
                require(row['has_history']==(i>0) and row['history_query']==(i-1 if i else None),'Causal query history')
            require(count==ep['executed_actions'],'Terminal action accounting')
            if ep['complete']:
                require(isinstance(ep['success'],bool),'Complete outcome type')
                if not ep['success']:require(count==(500 if m['task']=='place_shoe' else 400),'Failure not exhausted')
            else:require(ep['success'] is None,'Censored result counted')
            capture=arrays(run/'legal_inputs'/(ep['trace_stem']+'.npz'))
            require(set(capture)==FORWARD_KEYS|{'base'},'Illegal input capture')
            for i in range(min(2,len(rows))):
                for field in FORWARD_KEYS:require(array_hash(capture[field][i])==rows[i]['structured_input_sha256'][field],'Legal actual-input hash')
                require(array_hash(capture['base'][i,0])==array_hash(z['base'][i]),'Captured base')
                cell='A_bounded_clipped' if ep['condition']=='L0' else ep['condition']
                with torch.inference_mode():out=models[cell]({k:torch.from_numpy(v[i].copy()) for k,v in capture.items() if k!='base'},
                    torch.from_numpy(capture['base'][i].copy()),enabled=ep['condition']!='L0')
                require(np.max(np.abs(out['delta'][0].numpy()-z['raw_delta'][i]))<2e-5,'Live head output not reproducible')
            guard.observe(*key,rows[0],z['base'][0]);traces[key]=t
        except (ValueError,KeyError,OSError,RuntimeError,AssertionError) as error:errors.append(str(key)+': '+str(error))
    if set(traces)!=set(order):errors.append('Missing/extra valid branches')
    pairing=guard.summary(order)
    if not pairing['passed']:errors.append('First pairing incomplete')
    probe=_probe(run)
    if not probe['passed']:errors.append('Steady probe failed')
    outcomes={};changes={};repeats=[]
    for c in CONDITIONS:
        es=[e for e in episodes if e['condition']==c];full=[e for e in es if e['complete']]
        outcomes[c]=dict(successes=sum(e['success'] for e in full),completed=len(full),attempted=len(es),
            censored=len(es)-len(full),ratio=sum(e['success'] for e in full)/len(full) if full else None)
        if c!='L0':
            labels=[]
            for s,r,_ in order:
                if _!=c:continue
                a=traces.get((s,r,'L0'));b=traces.get((s,r,c))
                if not a or not b or not a['episode']['complete'] or not b['episode']['complete']:labels.append('censored');continue
                x,y=a['episode']['success'],b['episode']['success'];labels.append('tie' if x==y else 'rescued' if y else 'harmed')
            total=Counter(labels);changes[c]={k:total[k] for k in ('rescued','harmed','tie','censored')}
            changes[c]['net']=total['rescued']-total['harmed']
    if m['max_chunks'] is None:
        for s in m['seeds']:
            for c in CONDITIONS:
                a,b=traces.get((s,0,c)),traces.get((s,1,c))
                if a and b:
                    pair=_comparison(a,b,'full condition repeat');repeats.append(pair)
                    if not pair['passed']:errors.append('Full repeats differ: '+str((s,c)))
    full=m['max_chunks'] is None and len(episodes)==20 and all(e['complete'] for e in episodes)
    sources={str(run/f):digest(run/f) for f in ('manifest.json','episodes.json','steady_probe.json','steady_probe.npz')}
    for ep in episodes:
        for folder,suffix in (('traces','.npz'),('traces','.jsonl.gz'),('legal_inputs','.npz')):
            path=run/folder/(ep['trace_stem']+suffix)
            if path.exists():sources[str(path)]=digest(path)
    return dict(stage='output_support',task=m['task'],run=str(run),errors=errors,interface_passed=not errors,
        gate_passed=not errors and full,full_rollouts_complete=full,steady_probe=probe,initial_pairing=pairing,
        observed_outcomes=outcomes,changes=changes,within_condition_repeats=repeats,source_sha256=sources,
        scope='Two reused development scenes, two repeats; no population/generalization or training loss success claim')
