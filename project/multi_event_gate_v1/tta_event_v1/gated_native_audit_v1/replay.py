"""CPU replay of every actual gate candidate, decision, fast fallback and action."""
import inspect,types
from pathlib import Path
import numpy as np
from native_visible_expand_c_v1.analysis import branch as old_branch,arrays,same_query as old_same
from native_seven_task_gate_expansion_v1.paired_heads import forward,combine,ARM
from native_seven_task_gate_expansion_v1.gated_compose import EventComposer,FeedbackApplication
from native_seven_task_gate_expansion_v1.contract import read

def gate_record(policy):
    v=policy['gate_policy'];assert v['query']==policy['query'] and v['tick']==policy['actor_observation_tick']
    assert v['same_once_base_policy_file']==policy['policy_file']and v['same_once_base_policy_sha256']==policy['policy_sha256']and v['additional_VLA_or_noise_calls']==0
    assert read(Path(v['gate_readout_file']).with_suffix('.json'))==v
    return arrays(v['gate_readout_file'],v['gate_readout_sha256'])

# Keep all original native command receipts, phase95+lag, issuing-query,
# source, release and once-noise checks. Only the frozen candidate reference
# now comes from the explicitly recorded gate readout, rather than the aux head.
_before=inspect.getsource(old_branch)
_old="assert np.array_equal(action['base'],proposal['base'])and np.array_equal(action['reference'],proposal['raw_delta'])"
_new="assert np.array_equal(action['base'],proposal['base'])and np.array_equal(action['reference'],_gate_record(policy)['raw_delta'])"
assert _before.count(_old)==1
_source=_before.replace(_old,_new);_ns=dict(old_branch.__globals__,_gate_record=gate_record)
exec(compile(_source,__file__,'exec'),_ns)
branch=_ns['branch']

def same_query(a,b,**kwargs):
    old_same(a,b,**kwargs)
    x,y=gate_record(a['policy']),gate_record(b['policy'])
    assert x.keys()==y.keys()and all(x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes()for k in x)

def replay(result,spec,gate,aux):
    from cx_tta.multi_event import EventConfig
    composer=EventComposer(EventConfig(**spec['event_config']));application=FeedbackApplication()
    count=0;gate_scores=0;accepted_extra=0;rejected_extra=0
    for query in result['queries']:
        policy=query['policy'];old=arrays(policy['policy_file'],policy['policy_sha256']);g=gate_record(policy)
        legal={k.removeprefix('legal_'):v.copy()for k,v in old.items()if k.startswith('legal_')};base=legal.pop('base')[0]
        original_aux=forward(aux,legal,base);original_gate=forward(gate,legal,base)
        assert np.array_equal(base,g['base'])and np.array_equal(base,old['base'])
        for k in('raw_delta','hidden','logits'):
            assert np.array_equal(old[k],original_aux[k]),(query['query'],'aux',k)
            assert np.array_equal(g[k],original_gate[k]),(query['query'],'gate',k)
        action=arrays(query['actions_file'],query['actions_sha256'])
        assert np.array_equal(action['reference'],g['raw_delta'])
        if result['condition']=='baseline':
            assert np.array_equal(action['actions'],base)and not np.any(action['applied_delta']);count+=1;continue
        assert np.array_equal(action['auxiliary_reference'],old['raw_delta'])
        direction=None if result['condition']=='M_event_full'else action['auxiliary_direction']
        mapped,transfer=combine(g['raw_delta'],old['raw_delta'],direction)
        candidate=g['raw_delta'].copy()
        if mapped is not None:candidate[:,ARM]=mapped
        assert np.array_equal(candidate,action['gate_candidate'])and transfer==query['transferred_fast_increment']
        current=dict(legal,base=base[None].copy())
        aa,dd,event=composer.step(base,g['raw_delta'],current,adapted_ARM=mapped)
        aa,dd,event,receipt=application.step(composer,base,g['raw_delta'],mapped,aa,dd,event,query['response'],condition=result['condition'])
        assert np.array_equal(aa,action['actions'])and np.array_equal(dd,action['applied_delta'])
        assert event==query['event']and receipt==query['feedback_application_receipt']
        assert np.array_equal(aa[:,[6,13]],base[:,[6,13]])and np.max(np.abs(dd))<=.050001
        gate_scores+=len(event['all_event_gate_records'])
        accepted_extra+=bool(receipt.get('applied')and receipt.get('gate_decision'))
        rejected_extra+=receipt.get('reason')=='frozen_v118_rejected_fast_difference'
        count+=1
    if result['condition']!='baseline':assert result['feedback_application']==application.finish()
    return dict(passed=True,queries=count,original_gate_scoring_records_replayed=gate_scores,accepted_gate_approved_feedback_extra=accepted_extra,rejected_feedback_extra=rejected_extra,exact_dual_head_candidates_gate_context_scores_budget_cooldown_and_actions=True)
