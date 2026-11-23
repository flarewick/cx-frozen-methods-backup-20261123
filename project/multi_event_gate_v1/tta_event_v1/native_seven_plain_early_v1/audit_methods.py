"""Explicit audit dispatch for two distinct frozen candidate families."""
from native_visible_expand_c_v1.analysis import branch as plain_branch,same_query as plain_same,arrays
from gated_arm_context_v1.replay import branch as gate_branch,same_query as gate_same,replay

def branch(folder,condition,spec):
    return(gate_branch if spec['group']=='gated'else plain_branch)(folder,condition,spec)

def same_query(a,b,**kwargs):
    gate='gate_policy'in a['policy'];assert gate==('gate_policy'in b['policy'])
    return(gate_same if gate else plain_same)(a,b,**kwargs)

def replay_function(spec):
    if spec['group']=='plain':return None
    assert spec['group']=='gated'
    import torch
    from native_seven_task_gate_expansion_v1.paired_heads import build
    with torch.random.fork_rng(devices=[]):gate,aux,_,_=build()
    return lambda result:replay(result,spec,gate,aux)
