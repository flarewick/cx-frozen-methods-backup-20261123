"""Saved-number audit of causal decisions, identity and paired native outcomes."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml
from .audit import digest
from .native_history_cache import read,arrays,require,verify
from .structured_attribution_analysis import _load_trace,_comparison,_probe,_hash_json
from .causal_arm_gate import CONDITIONS,SMOKE_CONDITIONS
from .structured_action_adapter import FORWARD_KEYS,ARM_INDICES
from .output_support_adapter import OutputSupportAdapter,validate_native_arms
from .train_output_support import read_checkpoint
from .eval_diagnostics import array_hash


def order(seeds,conditions,block):
    return [(seed,block,c) for i,seed in enumerate(seeds) for c in (conditions if (i+block)%2==0 else tuple(reversed(conditions)))]


def summarize(run):
    run=Path(run);m=read(run/'manifest.json');p=read(m['protocol']);verify(p['source_sha256'])
    require(m['stage']==p['stage']=='causal_arm_confirmation' and digest(Path(m['protocol']))==m['protocol_sha256'],'Protocol identity')
    smoke=m['max_chunks'] is not None;conditions=SMOKE_CONDITIONS if smoke else CONDITIONS
    require(m['conditions']==list(conditions),'Registered conditions')
    if not smoke:
        reg=read(Path(m['registry'])/'records.json');require(digest(Path(m['registry'])/'records.json')==m['registry_records_sha256'],'Fixed registry')
        seeds=[s for s in p['seeds'] if next(r for r in reg if r['task']==m['task'] and r['seed']==s)['eligible']]
        require(m['attempted_seeds']==p['seeds'] and m['seeds']==seeds,'No scene replacement')
    expected=order(m['seeds'],conditions,m['block_id']);eps=read(run/'episodes.json');require(len(eps)==len(expected),'Expected paired panel')
    require([dict(seed=e['seed'],repeat=e['repeat'],condition=e['condition']) for e in eps]==m['order']==[dict(seed=s,repeat=b,condition=c) for s,b,c in expected],'Order/block contract')
    cfg=run/'robotwin/task_config/_eval_step_limit.yml';budget=yaml.safe_load(cfg.read_text())[m['task']]
    require(digest(cfg)==p['budget_config_sha256'][m['task']],'Original native budget config')
    hw=read(run/'hardware.json');require(hw['compiled'] and hw['shared_sampler'] and not hw['global_dynamo_disabled'] and hw['sample_kwargs']=={},'Official shared sampler')
    payload=read_checkpoint(Path(p['training_run'])/'A_bounded_clipped_seed0.pt','A_bounded_clipped')
    model=OutputSupportAdapter(32,payload['metadata']['geometry_mean'],'A_bounded_clipped').eval().requires_grad_(False);model.load_state_dict(payload['model'])
    errors=[];traces={};sources={};reload=[];active_count=off_count=0;events=[]
    for ep in eps:
        key=ep['seed'],ep['repeat'],ep['condition']
        try:
            t=_load_trace(run,ep);rows,z=t['rows'],t['arrays'];require(len(rows)>0 and len(rows)==ep['queries'],'Query contract')
            for k,a in z.items():require(a.shape==((len(rows),14) if k=='state' else (len(rows),32,14)) and a.dtype==np.float32 and np.isfinite(a).all(),'Numeric shape/type')
            count=0;first=None;window=None
            for i,row in enumerate(rows):
                b,raw,d=z['base'][i],z['raw_delta'][i],z['applied_delta'][i]
                decrease=z['state'][i,[6,13]]-np.clip(b[:,[6,13]],0,1).min(axis=0)
                event=bool(np.any(decrease>=.1));new=first is None and event
                if new:first=i;window=[i+1,i+3]
                eligible=window is not None and window[0]<=i<window[1]
                decision=dict(query=i,closing_decrease=decrease.tolist(),event_now=event,new_event=bool(new),first_event_query=first,
                    window=None if window is None else window.copy(),active=bool(eligible),state_sha256=array_hash(z['state'][i]),base_sha256=array_hash(b),threshold=.1)
                require(row['gate_decision']==decision,'Causal decision does not use exactly current fields')
                selected=m['fixed_window'] if ep['condition']=='A_fixed_arm' else window
                on=ep['condition']!='Z_zero' and selected is not None and selected[0]<=i<selected[1]
                require(row['window_active'] is bool(on) and row['applied_window']==selected and row['arm_strength']==float(on) and row['gripper_strength']==0,'Actual applied causal schedule')
                require(row['query']==i and row['evaluated_condition']==ep['condition'] and row['derived_checkpoint_sha256']==p['checkpoint_sha256'],'Frozen actor identity')
                require(row['has_history']==(i>0) and row['history_query']==(i-1 if i else None) and set(row['structured_input_sha256'])==FORWARD_KEYS,'Only legal/causal model fields')
                require(row['rng_before_query']==row['rng_after_feature'] and row['rng_before_query']!=row['rng_after_sampler'],'Internal stochastic query/noninterference')
                require(row['base_infer_calls']==row['auxiliary_prefix_calls']==1 and row['obs_updates']==32,'Official query/observations')
                require(row['base_chunk_sha256']==array_hash(b) and row['raw_delta_sha256']==array_hash(raw) and row['applied_delta_sha256']==array_hash(d),'Actual proposal/base/delta hashes')
                require(np.max(np.abs(raw)-np.asarray(p['scales']))<=1e-6,'Original proposal caps')
                want=np.zeros_like(raw);actions=b.copy()
                if on:want[:,ARM_INDICES]=raw[:,ARM_INDICES];actions[:,ARM_INDICES]=b[:,ARM_INDICES]+raw[:,ARM_INDICES];active_count+=1
                else:off_count+=1
                require(want.tobytes()==d.tobytes() and row['action_chunk_sha256']==array_hash(actions),'Actual arm-only action formula')
                require(actions[:,[6,13]].tobytes()==b[:,[6,13]].tobytes(),'Gripper commands not original base')
                require(row['robot_domain']==validate_native_arms(actions,p['robot_joint_limits']),'Domain changed')
                for which in ('before','after'):
                    state=row['native_'+which];require(state['usable'] and _hash_json(state)==row['native_'+which+'_sha256'],'Physical/state hash')
                    require(state['control']['step_lim']==budget,'Budget changed')
                before,after=row['native_before']['control'],row['native_after']['control']
                require(not before['eval_success'] and before['take_action_cnt']==count and after['take_action_cnt']==count+row['executed'] and 0<row['executed']<=32,'Original native clock')
                count=after['take_action_cnt'];require(count<=budget,'Native clock overflow')
            require(count==ep['executed_actions'],'Terminal clock accounting')
            if ep['complete']:
                require(type(ep['success']) is bool and ep['success']==after['eval_success'] and (ep['success'] or count==budget),'Complete terminal contract')
            else:require(ep['success'] is None,'Censored outcome counted')
            cap=arrays(run/'legal_inputs'/(ep['trace_stem']+'.npz'));ids=cap.pop('query_indices').tolist();require(set(cap)==FORWARD_KEYS|{'base'} and len(ids)<=7,'Small legal input capture')
            for j,i in enumerate(ids):
                require(all(array_hash(v[j])==rows[i]['structured_input_sha256'][k] for k,v in cap.items() if k!='base'),'Actual captured model fields')
                require(cap['base'][j,0].tobytes()==z['base'][i].tobytes(),'Captured actual base')
                with torch.inference_mode():out=model({k:torch.from_numpy(v[j].copy()) for k,v in cap.items() if k!='base'},torch.from_numpy(cap['base'][j].copy()),enabled=True)
                e=float(np.abs(out['delta'][0].numpy()-z['raw_delta'][i]).max());require(e<2e-5,'Frozen proposal reload');reload.append(e)
            events.append(dict(seed=ep['seed'],condition=ep['condition'],first_event=first,window=window,
                trigger_absent=first is None,window_started=window is not None and len(rows)>window[0],native_actions=count))
            traces[key]=t
            for folder,suffix in (('traces','.npz'),('traces','.jsonl.gz'),('legal_inputs','.npz')):
                file=run/folder/(ep['trace_stem']+suffix);sources[str(file)]=digest(file)
        except (ValueError,OSError,KeyError,RuntimeError,AssertionError) as exc:errors.append(str(key)+': '+str(exc))
    if set(traces)!=set(expected):errors.append('Missing/invalid branches')
    probe=_probe(run)
    if not probe['passed']:errors.append('TrueL0 common-zero fixed-input probe failed')
    pairs=[];fixed_pairs=[]
    for s,b,c in expected:
        if c=='Z_zero':continue
        a,t=traces.get((s,b,'Z_zero')),traces.get((s,b,c))
        if a is None or t is None:continue
        w=m['fixed_window'] if c=='A_fixed_arm' else t['rows'][-1]['gate_decision']['window']
        shared=min(len(a['rows']),len(t['rows']),w[0]+1 if w else len(a['rows']));ok=shared>0
        for i in range(shared):
            x,y=a['rows'][i],t['rows'][i]
            fields=('observation_sha256','native_before_sha256','rng_before_query','rng_after_sampler','structured_input_sha256','base_chunk_sha256','raw_delta_sha256')
            ok=ok and all(x[k]==y[k] for k in fields)
            if w is None or i<w[0]:ok=ok and x['native_after_sha256']==y['native_after_sha256'] and x['action_chunk_sha256']==y['action_chunk_sha256']
        if not ok:errors.append('Causal prefix/fork pairing failed '+str((s,b,c)))
        pairs.append(dict(seed=s,condition=c,shared_pre_query_states=shared,passed=ok))
    if smoke:
        for s in m['seeds']:
            a,b=traces.get((s,m['block_id'],'A_fixed_arm')),traces.get((s,m['block_id'],'A_causal_arm'))
            if a and b:
                cmp=_comparison(a,b,'Fixed-window vs causal-window interface equivalence')
                cmp['numeric_exact']=all(a['arrays'][k].tobytes()==b['arrays'][k].tobytes() for k in a['arrays'])
                cmp['passed']=cmp['passed'] and cmp['numeric_exact'];fixed_pairs.append(cmp)
                if not cmp['passed']:errors.append('Known fixed-window and causal window differ physically')
    outcomes={};changes={}
    for c in conditions:
        es=[e for e in eps if e['condition']==c];full=[e for e in es if e['complete']]
        outcomes[c]=dict(successes=sum(e['success'] for e in full),completed=len(full),censored=len(es)-len(full),ratio=sum(e['success'] for e in full)/len(full) if full else None)
        if c!='Z_zero':
            labels=Counter()
            for s in m['seeds']:
                a,b=traces.get((s,m['block_id'],'Z_zero')),traces.get((s,m['block_id'],c))
                if a and b and a['episode']['complete'] and b['episode']['complete']:
                    x,y=a['episode']['success'],b['episode']['success'];labels['tie' if x==y else 'rescued' if y else 'harmed']+=1
                else:labels['censored']+=1
            changes[c]={k:labels[k] for k in ('rescued','harmed','tie','censored')};changes[c]['net']=labels['rescued']-labels['harmed']
    full=not smoke and bool(m['seeds']) and all(e['complete'] for e in eps)
    for name in ('manifest.json','episodes.json','steady_probe.json','steady_probe.npz','hardware.json'):sources[str(run/name)]=digest(run/name)
    return dict(stage=p['stage'],run=str(run),task=m['task'],block_id=m['block_id'],process_instance_id=m['process_instance_id'],
        interface_passed=not errors,gate_passed=not errors and full,full_rollouts_complete=full,errors=errors,
        observed_outcomes=outcomes,changes=changes,events=events,causal_prefix_pairs=pairs,fixed_vs_causal_pairs=fixed_pairs,
        active_queries=active_count,zero_identity_queries=off_count,actual_inputs_checked=len(reload),CPU_proposal_max_error=max(reload,default=None),
        native_budget=budget,source_sha256=sources,scope='One independent compiled-process block, paired within block; heuristic causal timing baseline,not learned router or majority-task claim')
