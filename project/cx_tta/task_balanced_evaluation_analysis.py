"""Independent compact-array audit for the S2l three-condition repeated test."""
from collections import Counter
from pathlib import Path

import numpy as np

from .audit import digest
from .eval_diagnostics import array_hash
from .structured_attribution_analysis import _comparison, _load_trace, _probe, _trace_errors
from .train_structured_action import read

CONDITIONS=('L0','S0','S_task_balanced')


def evaluation_order(seeds, *, smoke=False):
    if len(seeds)!=2 or len(set(seeds))!=2:
        raise ValueError('Two distinct preregistered scenes required')
    if smoke:
        return [(seeds[0],0,c) for c in CONDITIONS]
    return [(seed,repeat,condition) for index,seed in enumerate(seeds) for repeat in range(2)
        for condition in (CONDITIONS if (index+repeat)%2==0 else tuple(reversed(CONDITIONS)))]


def chunk_number_errors(trace):
    """Bind saved deltas, unit-strength application and action hashes exactly."""
    errors=[];arrays=trace['arrays'];condition=trace['episode']['condition']
    for i,row in enumerate(trace['rows']):
        base,raw,applied=(arrays[key][i] for key in ('base','raw_delta','applied_delta'))
        if row.get('raw_delta_sha256')!=array_hash(raw):errors.append(f'query{i}: raw delta hash mismatch')
        if row.get('applied_delta_sha256')!=array_hash(applied):errors.append(f'query{i}: applied delta hash mismatch')
        if condition=='L0':
            if np.any(raw) or np.any(applied):errors.append(f'query{i}: L0 computed nonzero delta')
            actions=base
        else:
            if array_hash(raw)!=array_hash(applied):errors.append(f'query{i}: full-strength raw/applied mismatch')
            actions=base+applied
        if row.get('action_chunk_sha256')!=array_hash(actions):errors.append(f'query{i}: reconstructed action hash mismatch')
    return errors


class InitialPairingGuard:
    def __init__(self):
        self.initial={};self.comparisons={}

    def observe(self,seed,repeat,condition,row,base):
        key=(seed,repeat,condition)
        if key in self.initial:raise RuntimeError('Duplicate initial query')
        if condition not in CONDITIONS:raise ValueError('Unknown condition')
        self.initial[key]=(dict(row),np.asarray(base,np.float32).copy())
        reference=self.initial.get((seed,repeat,'L0'))
        if reference is None:return
        left,left_base=reference
        for candidate in CONDITIONS[1:]:
            current=self.initial.get((seed,repeat,candidate));pair=(seed,repeat,candidate)
            if current is None or pair in self.comparisons:continue
            right,right_base=current
            checks={field:bool(left.get(field) and left[field]==right.get(field))
                for field in ('observation_sha256','native_before_sha256','rng_before_query','rng_after_sampler')}
            checks['structured_inputs']=bool(left.get('structured_input_sha256') and left['structured_input_sha256']==right.get('structured_input_sha256'))
            checks['reset_history']=left.get('has_history') is False and right.get('has_history') is False and left.get('history_query') is None and right.get('history_query') is None
            checks['base_exact']=array_hash(left_base)==array_hash(right_base)
            difference=left_base.astype(np.float64)-right_base.astype(np.float64)
            result=dict(seed=seed,repeat=repeat,condition=candidate,checks=checks,
                max_abs=float(np.abs(difference).max()),rms=float(np.sqrt(np.mean(difference**2))),passed=all(checks.values()))
            self.comparisons[pair]=result
            if not result['passed']:raise RuntimeError('Initial query is not paired: '+str(result))

    def summary(self,order):
        expected={(s,r,c) for s,r,c in order if c!='L0'}
        measured=[self.comparisons[k] for k in sorted(self.comparisons)]
        return dict(expected_comparisons=len(expected),comparisons=measured,
            passed=set(self.comparisons)==expected and all(r['passed'] for r in measured))


def summarize(run):
    run=Path(run);manifest=read(run/'manifest.json');episodes=read(run/'episodes.json')
    expected=evaluation_order(manifest['seeds'],smoke=manifest['max_chunks'] is not None)
    errors=[]
    fixed={'stage':'task_balanced','conditions':list(CONDITIONS),'compiled':True,
        'policy_init_seed':0,'rollout_protocol':'official','noise_source':'model_internal_torch',
        'shared_base_models':1,'shared_compiled_samplers':1,'original_native_budget':True,'no_video':True,
        'task_config':'demo_randomized','instruction_type':'unseen'}
    for key,value in fixed.items():
        if type(manifest.get(key)) is not type(value) or manifest[key]!=value:errors.append('Invalid manifest '+key)
    protocol_path=Path(manifest['evaluation_protocol'])
    if digest(protocol_path)!=manifest['evaluation_protocol_sha256']:errors.append('Evaluation protocol changed')
    protocol=read(protocol_path)
    from .evaluate_task_balanced import validate_protocol
    accepted,_,_,_=validate_protocol(Path(protocol['training_run']),manifest['task'])
    if accepted!=protocol:errors.append('Validated protocol differs from stored protocol')
    from .train_structured_action import verify
    verify(protocol['source_sha256'])
    for condition,sha in protocol['checkpoints'].items():
        if manifest['checkpoint_sha256'].get(condition)!=sha:errors.append('Checkpoint identity differs: '+condition)
    if manifest['seeds']!=protocol['seeds'][manifest['task']]:errors.append('Scene selection differs')
    if manifest['order']!=[dict(seed=s,repeat=r,condition=c) for s,r,c in expected]:errors.append('Episode order differs')
    traces={};guard=InitialPairingGuard()
    for episode in episodes:
        key=(episode['seed'],episode['repeat'],episode['condition'])
        if key in traces:errors.append('Duplicate episode '+str(key));continue
        try:
            trace=_load_trace(run,episode);problems=_trace_errors(trace,sham=episode['condition']=='L0')+chunk_number_errors(trace)
            if problems:raise ValueError('; '.join(problems))
            if any(r.get('evaluated_condition')!=episode['condition'] for r in trace['rows']):
                raise ValueError('Trace/evaluated checkpoint condition mismatch')
            if episode['condition']=='S_task_balanced' and any(r.get('derived_checkpoint_sha256')!=protocol['checkpoints']['S_task_balanced'] for r in trace['rows']):
                raise ValueError('Trace derived checkpoint mismatch')
            if any(r.get('arm_strength')!=(0. if episode['condition']=='L0' else 1.)
                    or r.get('gripper_strength')!=(0. if episode['condition']=='L0' else 1.)
                    or r.get('history_policy')!='real' for r in trace['rows']):
                raise ValueError('Unregistered strength/history change')
            guard.observe(*key,trace['rows'][0],trace['arrays']['base'][0]);traces[key]=trace
        except (OSError,ValueError,KeyError,RuntimeError) as error:errors.append(str(key)+': '+str(error))
    if set(traces)!=set(expected):errors.append('Missing/extra/invalid expected traces')
    probe=_probe(run)
    if not probe['passed']:errors.append('Fixed-input probe failed')
    pairing=guard.summary(expected)
    if not pairing['passed']:errors.append('Initial pair coverage failed')
    outcomes={};changes={};pairs=[];repeats=[]
    for condition in CONDITIONS:
        rows=[r for r in episodes if r['condition']==condition]
        complete=[r for r in rows if r.get('complete') is True and type(r.get('success')) is bool]
        count=sum(r['success'] for r in complete)
        outcomes[condition]=dict(successes=count,completed=len(complete),attempted=len(rows),censored=len(rows)-len(complete),ratio=count/len(complete) if complete else None)
    contexts=sorted({(s,r) for s,r,_ in expected})
    for seed,repeat in contexts:
        baseline=traces.get((seed,repeat,'L0'))
        for condition in CONDITIONS[1:]:
            candidate=traces.get((seed,repeat,condition))
            if baseline is None or candidate is None:continue
            comparison=_comparison(baseline,candidate,'same_runtime_candidate_vs_L0')
            numeric=comparison['base_difference_same_observation_rng']
            if numeric['byte_exact'] is False:errors.append('Same observation/RNG has different base numbers: '+str((seed,repeat,condition)))
            complete=all(t['episode'].get('complete') is True and type(t['episode'].get('success')) is bool for t in (baseline,candidate))
            if not complete:label='censored'
            elif baseline['episode']['success']==candidate['episode']['success']:label='tie'
            else:label='rescued' if candidate['episode']['success'] else 'harmed'
            pairs.append(dict(comparison,repeat=repeat,condition=condition,outcome_change=label))
    for condition in CONDITIONS[1:]:
        counts=Counter(p['outcome_change'] for p in pairs if p['condition']==condition)
        changes[condition]={k:counts[k] for k in ('rescued','harmed','tie','censored')}
        changes[condition]['net']=counts['rescued']-counts['harmed']
    if manifest['max_chunks'] is None:
        for seed in manifest['seeds']:
            for condition in CONDITIONS:
                left,right=traces.get((seed,0,condition)),traces.get((seed,1,condition))
                if left is None or right is None:continue
                comparison=_comparison(left,right,'condition_repeat_0_vs_1')
                repeats.append(comparison)
                if comparison['base_difference_same_observation_rng']['byte_exact'] is False:
                    errors.append('Repeated equal observation/RNG base numbers differ: '+str((seed,condition)))
    full=(manifest['max_chunks'] is None and len(episodes)==12 and all(r.get('complete') is True and type(r.get('success')) is bool for r in episodes))
    sources={str(run/name):digest(run/name) for name in ('manifest.json','episodes.json','steady_probe.json','steady_probe.npz')}
    for episode in episodes:
        for suffix in ('.npz','.jsonl.gz'):
            path=run/'traces'/(episode['trace_stem']+suffix)
            if path.is_file():sources[str(path)]=digest(path)
    return dict(stage='task_balanced',task=manifest['task'],run=str(run),errors=errors,
        interface_passed=not errors,gate_passed=not errors and full,full_rollouts_complete=full,
        steady_probe=probe,initial_pairing=pairing,observed_outcomes=outcomes,changes=changes,
        pairs=pairs,within_condition_repeats=repeats,source_sha256=sources,
        scope='Two reused development scenes, two repetitions. No independent task-level generalization claim.')
