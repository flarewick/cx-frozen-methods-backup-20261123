"""Independent numeric/native-domain audit of a local arm/gripper factorial."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml
from .audit import digest
from .native_history_cache import read,arrays,require,verify
from .structured_attribution_analysis import _load_trace,_comparison,_probe,_hash_json
from .local_component_intervention import CONDITIONS,FACTORS,GRIPPERS
from .structured_action_adapter import FORWARD_KEYS,ARM_INDICES
from .output_support_adapter import OutputSupportAdapter,validate_native_arms
from .train_output_support import read_checkpoint
from .eval_diagnostics import array_hash


def episode_order(seeds,smoke=False):
    return [(s,r,c) for i,s in enumerate(seeds) for r in range(1 if smoke else 2)
            for c in (CONDITIONS if (i+r)%2==0 else tuple(reversed(CONDITIONS)))]


def summarize(run):
    run=Path(run);m=read(run/'manifest.json');p=read(Path(m['protocol']));verify(p['source_sha256'])
    require(m['stage']==p['stage']=='local_component_factorial','Correct stage')
    require(digest(Path(m['protocol']))==m['protocol_sha256'],'Sealed protocol identity')
    require(p['conditions']=={c:dict(arm_strength=a,gripper_strength=g) for c,(a,g) in FACTORS.items()},'Registered factors')
    require(m['conditions']==list(CONDITIONS) and m['window']==p['window']==[1,3],'Fixed window/cells')
    require(p['arm_indices']==list(ARM_INDICES) and p['gripper_indices']==list(GRIPPERS),'Original robot channel groups')
    require(m['seeds']==(p['seeds'][:1] if m['max_chunks'] is not None else p['seeds']),'Fixed scene selection')
    expected=episode_order(m['seeds'],m['max_chunks'] is not None);eps=read(run/'episodes.json')
    require(m['order']==[dict(seed=s,repeat=r,condition=c) for s,r,c in expected],'Declared order')
    require([dict(seed=e['seed'],repeat=e['repeat'],condition=e['condition']) for e in eps]==m['order'][:len(eps)],'Actual execution order')
    cfg=run/'robotwin/task_config/_eval_step_limit.yml';budget=yaml.safe_load(cfg.read_text())[m['task']]
    require(digest(cfg)==p['budget_config_sha256'] and type(budget) is int and budget>0,'Frozen original budget config')
    hw=read(run/'hardware.json');require(hw['compiled'] and hw['shared_sampler'] and not hw['global_dynamo_disabled'] and hw['sample_kwargs']=={},'Shared official sampler')
    payload=read_checkpoint(Path(p['training_run'])/'A_bounded_clipped_seed0.pt','A_bounded_clipped')
    model=OutputSupportAdapter(32,payload['metadata']['geometry_mean'],'A_bounded_clipped').eval().requires_grad_(False);model.load_state_dict(payload['model'])
    errors=[];traces={};sources={};readbacks=[];arm_count=grip_count=zero_count=window_count=0
    for ep in eps:
        key=ep['seed'],ep['repeat'],ep['condition']
        try:
            require(key in expected and key not in traces,'Unique registered episode')
            t=_load_trace(run,ep);rows,z=t['rows'],t['arrays'];require(len(rows)>0 and len(rows)==ep['queries'],'Query count')
            for f,a in z.items():
                require(a.shape==((len(rows),14) if f=='state' else (len(rows),32,14)) and a.dtype==np.float32 and np.isfinite(a).all(),'Native numeric contract')
            count=0
            for i,row in enumerate(rows):
                within=p['window'][0]<=i<p['window'][1]
                aa=p['conditions'][ep['condition']]['arm_strength'] if within else 0
                gg=p['conditions'][ep['condition']]['gripper_strength'] if within else 0
                b,raw,d=z['base'][i],z['raw_delta'][i],z['applied_delta'][i]
                require(row['query']==i and row['evaluated_condition']==ep['condition'],'Query/condition identity')
                require(row['within_window'] is within and row['window_active'] is bool(aa or gg) and row['window']==p['window'],'Actual time mask')
                require(row['component_factors']==list(FACTORS[ep['condition']]) and row['arm_strength']==aa and row['gripper_strength']==gg,'Actual component strengths')
                require(row['derived_checkpoint_sha256']==p['checkpoint_sha256'],'Frozen model identity')
                require(row['output_mode']=='bounded' and row['history_policy']=='real','Original bounded model and real history')
                require(row['has_history']==(i>0) and row['history_query']==(i-1 if i else None),'Causal native history')
                require(set(row['structured_input_sha256'])==FORWARD_KEYS,'Only9 model fields')
                require(row['base_infer_calls']==row['auxiliary_prefix_calls']==1 and row['obs_updates']==32,'One query/per-action observations')
                require(row['rng_before_query']==row['rng_after_feature'] and row['rng_before_query']!=row['rng_after_sampler'],'Model internal noise/noninterference')
                require(row['base_chunk_sha256']==array_hash(b) and row['raw_delta_sha256']==array_hash(raw) and row['applied_delta_sha256']==array_hash(d),'Actual numeric hashes')
                require(np.max(np.abs(raw)-np.asarray(p['scales']))<=1e-6,'Original proposal caps')
                want=np.zeros_like(raw);actions=b.copy()
                # Independent registered-column check, not the actor's helper.
                for factor,columns in ((aa,tuple(p['arm_indices'])),(gg,tuple(p['gripper_indices']))):
                    if factor:
                        want[:,columns]=raw[:,columns]
                        actions[:,columns]=b[:,columns]+raw[:,columns]
                    else:require(d[:,columns].tobytes()==np.zeros_like(d[:,columns]).tobytes(),'Disabled column not positive-zero')
                require(want.tobytes()==d.tobytes() and row['action_chunk_sha256']==array_hash(actions),'Exact actual component assembly')
                if not (aa or gg):require(actions.tobytes()==b.tobytes(),'Zero intervention not exact base');zero_count+=1
                arm_count+=aa;grip_count+=gg;window_count+=int(within)
                require(row['robot_domain']==validate_native_arms(actions,p['robot_joint_limits']),'Original native-domain accounting')
                for which in ('before','after'):
                    n=row['native_'+which];require(n['usable'] and _hash_json(n)==row['native_'+which+'_sha256'],'Native state/hash coverage')
                    require(n['control']['step_lim']==budget,'Original budget changed')
                before,after=row['native_before']['control'],row['native_after']['control']
                require(not before['eval_success'] and before['take_action_cnt']==count and 0<row['executed']<=32 and after['take_action_cnt']==count+row['executed'],'Native clock continuity')
                count=after['take_action_cnt'];require(count<=budget,'Budget overflow')
            require(count==ep['executed_actions'],'Terminal action accounting')
            if ep['complete']:
                require(type(ep['success']) is bool and ep['success']==after['eval_success'],'Actual terminal flag')
                require(ep['success'] or count==budget,'Premature failure counted complete')
            else:require(ep['success'] is None,'Censored record given outcome')
            cap=arrays(run/'legal_inputs'/(ep['trace_stem']+'.npz'));indices=cap.pop('query_indices').tolist()
            require(set(cap)==FORWARD_KEYS|{'base'} and indices==sorted(set(indices)) and len(indices)<=6,'Registered small legal capture')
            for j,i in enumerate(indices):
                require(all(array_hash(v[j])==rows[i]['structured_input_sha256'][k] for k,v in cap.items() if k!='base'),'Captured actual model inputs')
                require(cap['base'][j,0].tobytes()==z['base'][i].tobytes(),'Captured base command')
                with torch.inference_mode():out=model({k:torch.from_numpy(v[j].copy()) for k,v in cap.items() if k!='base'},torch.from_numpy(cap['base'][j].copy()),enabled=True)
                error=float(np.max(np.abs(out['delta'][0].numpy()-z['raw_delta'][i])));require(error<2e-5,'Actual shared proposal not reproducible');readbacks.append(error)
            traces[key]=t
            for folder,suffix in (('traces','.npz'),('traces','.jsonl.gz'),('legal_inputs','.npz')):
                path=run/folder/(ep['trace_stem']+suffix);sources[str(path)]=digest(path)
        except (OSError,ValueError,KeyError,AssertionError,RuntimeError) as exc:errors.append(str(key)+': '+str(exc))
    if set(traces)!=set(expected):errors.append('Missing/invalid branches')
    probe=_probe(run)
    if not probe['passed']:errors.append('TrueL0/common-zero proposal probe failed')
    pairs=[];repeated=[]
    for s,r,c in expected:
        if c=='C00_zero':continue
        a,b=traces.get((s,r,'C00_zero')),traces.get((s,r,c))
        if a is None or b is None:continue
        shared=min(p['window'][0]+1,len(a['rows']),len(b['rows']));passed=shared>0
        for i in range(shared):
            x,y=a['rows'][i],b['rows'][i]
            passed=passed and all(x[f]==y[f] for f in ('observation_sha256','native_before_sha256','rng_before_query','rng_after_sampler','structured_input_sha256','base_chunk_sha256','raw_delta_sha256'))
            if i<p['window'][0]:passed=passed and x['action_chunk_sha256']==y['action_chunk_sha256'] and x['native_after_sha256']==y['native_after_sha256']
        if not passed:errors.append('Matched raw candidate/base/window-entry failed '+str((s,r,c)))
        pairs.append(dict(seed=s,repeat=r,condition=c,shared_pre_query_states=shared,passed=passed))
    if m['max_chunks'] is None:
        for s in m['seeds']:
            for c in CONDITIONS:
                a,b=traces.get((s,0,c)),traces.get((s,1,c))
                if a is not None and b is not None:
                    row=_comparison(a,b,'same component schedule repeat')
                    row['all_raw_and_applied_bytes_equal']=all(a['arrays'][k].tobytes()==b['arrays'][k].tobytes() for k in ('base','raw_delta','applied_delta','state'))
                    row['passed']=row['passed'] and row['all_raw_and_applied_bytes_equal'];repeated.append(row)
                    if not row['passed']:errors.append('Component repeats differ '+str((s,c)))
    outcomes={};changes={}
    for c in CONDITIONS:
        es=[e for e in eps if e['condition']==c];full=[e for e in es if e['complete']]
        outcomes[c]=dict(successes=sum(e['success'] for e in full),completed=len(full),censored=len(es)-len(full),ratio=sum(e['success'] for e in full)/len(full) if full else None)
        if c!='C00_zero':
            labels=Counter()
            for s,r,cell in expected:
                if cell!=c:continue
                a,b=traces.get((s,r,'C00_zero')),traces.get((s,r,c))
                if a and b and a['episode']['complete'] and b['episode']['complete']:
                    x,y=a['episode']['success'],b['episode']['success'];labels['tie' if x==y else 'rescued' if y else 'harmed']+=1
                else:labels['censored']+=1
            changes[c]={k:labels[k] for k in ('rescued','harmed','tie','censored')};changes[c]['net']=labels['rescued']-labels['harmed']
    full=m['max_chunks'] is None and len(eps)==len(expected) and all(e['complete'] for e in eps)
    for name in ('manifest.json','episodes.json','steady_probe.json','steady_probe.npz','hardware.json'):
        sources[str(run/name)]=digest(run/name)
    return dict(stage='local_component_factorial',run=str(run),task=m['task'],interface_passed=not errors,gate_passed=not errors and full,
        full_rollouts_complete=full,errors=errors,native_budget=budget,observed_outcomes=outcomes,changes=changes,
        proposal_and_window_prefix_pairs=pairs,within_condition_repeats=repeated,
        arm_queries_checked=arm_count,gripper_queries_checked=grip_count,window_queries_checked=window_count,
        zero_identity_queries_checked=zero_count,actual_inputs_checked=len(readbacks),CPU_reload_max_error=max(readbacks,default=None),
        steady_probe_passed=probe['passed'],source_sha256=sources,
        scope='Two reused cup development scenes; fixed local arm/gripper interaction, not learned router or population success evidence')
