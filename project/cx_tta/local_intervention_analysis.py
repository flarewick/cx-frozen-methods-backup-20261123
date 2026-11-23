"""Read saved local masks, verify shared prefixes and original native budgets."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml
from .audit import digest
from .native_history_cache import read,arrays,require,verify
from .structured_attribution_analysis import _load_trace,_comparison,_probe,_hash_json
from .local_intervention import CONDITIONS,active_at
from .output_support_adapter import OutputSupportAdapter,validate_native_arms
from .train_output_support import read_checkpoint
from .eval_diagnostics import array_hash
from .structured_action_adapter import FORWARD_KEYS


def order(seeds,smoke=False):
    return [(s,r,c) for i,s in enumerate(seeds) for r in range(1 if smoke else 2)
            for c in (CONDITIONS if (i+r)%2==0 else tuple(reversed(CONDITIONS)))]


def summarize(run):
    run=Path(run);m=read(run/'manifest.json');p=read(Path(m['protocol']));verify(p['source_sha256'])
    require(digest(Path(m['protocol']))==m['protocol_sha256'],'Sealed local protocol')
    eps=read(run/'episodes.json');expected=order(m['seeds'],m['max_chunks'] is not None)
    budget=yaml.safe_load((run/'robotwin/task_config/_eval_step_limit.yml').read_text())[m['task']]
    require(digest(run/'robotwin/task_config/_eval_step_limit.yml')==p['budget_config_sha256'][m['task']],'Original budget source')
    payload=read_checkpoint(Path(p['training_run'])/'A_bounded_clipped_seed0.pt','A_bounded_clipped')
    model=OutputSupportAdapter(32,payload['metadata']['geometry_mean'],'A_bounded_clipped').eval().requires_grad_(False)
    model.load_state_dict(payload['model']);traces={};errors=[];sources={};readbacks=[];active=off=0
    for ep in eps:
        key=(ep['seed'],ep['repeat'],ep['condition'])
        try:
            require(key in expected and key not in traces,'Unique registered episode')
            t=_load_trace(run,ep);rows=t['rows'];z=t['arrays'];w=p['windows'][m['task']][str(ep['seed'])]
            count=0
            for name,a in z.items():
                require(a.shape==((len(rows),14) if name=='state' else (len(rows),32,14)) and a.dtype==np.float32 and np.isfinite(a).all(),'Numeric contract')
            for i,row in enumerate(rows):
                enabled=active_at(ep['condition'],i,w);b,d,raw=z['base'][i],z['applied_delta'][i],z['raw_delta'][i]
                require(row['query']==i and row['evaluated_condition']==ep['condition'],'Query/cell identity')
                require(row['window_active'] is enabled and row['arm_strength']==row['gripper_strength']==float(enabled),'Actual schedule')
                require(row['window_schedule']==w,'Window changed')
                require(row['derived_checkpoint_sha256']==p['checkpoint_sha256'],'Same frozenA checkpoint')
                require(row['has_history']==(i>0) and row['history_query']==(i-1 if i else None),'Native causal history')
                require(row['rng_before_query']==row['rng_after_feature'] and row['rng_before_query']!=row['rng_after_sampler'],'Sampler/noise contract')
                require(row['base_infer_calls']==row['auxiliary_prefix_calls']==1 and row['obs_updates']==32,'One query/per-action observations')
                require(row['base_chunk_sha256']==array_hash(b) and row['raw_delta_sha256']==array_hash(raw) and row['applied_delta_sha256']==array_hash(d),'Stored number hashes')
                require(np.max(np.abs(raw)-np.asarray(p['scales']))<=1e-6,'Original bounded proposal')
                if enabled:require(raw.tobytes()==d.tobytes(),'Active window changed strength');active+=1
                else:require(not np.any(d),'Window off did not suppress residual');off+=1
                actions=b+d if enabled else b
                require(array_hash(actions)==row['action_chunk_sha256'],'Actual command formula')
                require(row['robot_domain']==validate_native_arms(actions,p['robot_joint_limits']),'Native robot domain')
                for which in ('before','after'):
                    require(row['native_'+which]['usable'] and _hash_json(row['native_'+which])==row['native_'+which+'_sha256'],'Physical hash')
                    require(row['native_'+which]['control']['step_lim']==budget,'Native budget unchanged')
                before,after=row['native_before']['control'],row['native_after']['control']
                require(before['take_action_cnt']==count and after['take_action_cnt']==count+row['executed'],'Contiguous original clock')
                require(not before['eval_success'] and 0<row['executed']<=32,'No query after success')
                count=after['take_action_cnt']
            require(len(rows)==ep['queries'] and count==ep['executed_actions'],'Terminal accounting')
            if ep['complete']:
                require(type(ep['success']) is bool and ep['success']==after['eval_success'],'Actual success')
                require(ep['success'] or count==budget,'Complete failure exhausted original budget')
            else:require(ep['success'] is None,'Censored outcome miscounted')
            cap=arrays(run/'legal_inputs'/(ep['trace_stem']+'.npz'));indices=cap.pop('query_indices').tolist()
            require(set(cap)==FORWARD_KEYS|{'base'},'Exactly legal input capture')
            for j,i in enumerate(indices):
                require(all(array_hash(v[j])==rows[i]['structured_input_sha256'][k] for k,v in cap.items() if k!='base'),'Actual packet hashes')
                require(cap['base'][j,0].tobytes()==z['base'][i].tobytes(),'Actual base packet')
                with torch.inference_mode():
                    out=model({k:torch.from_numpy(v[j].copy()) for k,v in cap.items() if k!='base'},torch.from_numpy(cap['base'][j].copy()),enabled=ep['condition']!='L0')
                error=float(np.max(np.abs(out['delta'][0].numpy()-z['raw_delta'][i])));require(error<2e-5,'Actual proposal reload');readbacks.append(error)
            traces[key]=t
            for folder,suffix in (('traces','.npz'),('traces','.jsonl.gz'),('legal_inputs','.npz')):
                path=run/folder/(ep['trace_stem']+suffix);sources[str(path)]=digest(path)
        except (OSError,ValueError,KeyError,AssertionError,RuntimeError) as exc:errors.append(str(key)+': '+str(exc))
    if set(traces)!=set(expected):errors.append('Missing/invalid branches')
    probe=_probe(run)
    if not probe['passed']:errors.append('Common fixed-input zero probe failed')
    pairings=[];repeated=[]
    for s,r,c in expected:
        if c=='L0':continue
        a,b=traces.get((s,r,'L0')),traces.get((s,r,c))
        if a is None or b is None:continue
        start=0 if c=='A_always' else p['windows'][m['task']][str(s)][c][0]
        shared=min(start+1,len(a['rows']),len(b['rows']))
        checks=[]
        for i in range(shared):
            x,y=a['rows'][i],b['rows'][i]
            fields=('observation_sha256','native_before_sha256','rng_before_query','rng_after_sampler','structured_input_sha256')
            ok=all(x[f]==y[f] for f in fields) and a['arrays']['base'][i].tobytes()==b['arrays']['base'][i].tobytes()
            if i<start:ok=ok and x['action_chunk_sha256']==y['action_chunk_sha256'] and x['native_after_sha256']==y['native_after_sha256']
            checks.append(ok)
        passed=bool(checks) and all(checks)
        if not passed:errors.append('Window prefix/fork pairing failed '+str((s,r,c)))
        pairings.append(dict(seed=s,repeat=r,condition=c,shared_pre_query_states=shared,start_query=start,passed=passed))
    if m['max_chunks'] is None:
        for seed in m['seeds']:
            for c in CONDITIONS:
                a,b=traces.get((seed,0,c)),traces.get((seed,1,c))
                if a is not None and b is not None:
                    row=_comparison(a,b,'same local schedule repeat');repeated.append(row)
                    if not row['passed']:errors.append('Within-schedule repeats differ '+str((seed,c)))
    outcomes={};changes={}
    for c in CONDITIONS:
        es=[e for e in eps if e['condition']==c];full=[e for e in es if e['complete']]
        outcomes[c]=dict(successes=sum(e['success'] for e in full),completed=len(full),censored=len(es)-len(full),ratio=sum(e['success'] for e in full)/len(full) if full else None)
        if c!='L0':
            counts=Counter()
            for s,r,cell in expected:
                if cell!=c:continue
                a,b=traces.get((s,r,'L0')),traces.get((s,r,c))
                if a and b and a['episode']['complete'] and b['episode']['complete']:
                    x,y=a['episode']['success'],b['episode']['success'];counts['tie' if x==y else 'rescued' if y else 'harmed']+=1
                else:counts['censored']+=1
            changes[c]={k:counts[k] for k in ('rescued','harmed','tie','censored')}
            changes[c]['net']=counts['rescued']-counts['harmed']
    full=m['max_chunks'] is None and len(eps)==len(expected) and all(e['complete'] for e in eps)
    for name in ('manifest.json','episodes.json','steady_probe.json','steady_probe.npz'):
        sources[str(run/name)]=digest(run/name)
    return dict(stage='local_intervention',task=m['task'],run=str(run),errors=errors,interface_passed=not errors,
        full_rollouts_complete=full,gate_passed=not errors and full,native_budget=budget,
        observed_outcomes=outcomes,changes=changes,window_prefix_pairings=pairings,within_condition_repeats=repeated,
        active_queries_checked=active,off_queries_identity_checked=off,actual_inputs_checked=len(readbacks),CPU_reload_max_error=max(readbacks,default=None),
        steady_probe_passed=probe['passed'],source_sha256=sources,
        scope='Fixed diagnostic windows from prior L0; no learned gate or population generalization claim')
