"""Real cached input verification of the two-head actor and explicit episode edits."""
import ast,json
from pathlib import Path
import numpy as np
from .contract import T,O,GATE_TRAINING,read,mapped,sha,plan

def main():
    import torch
    from .paired_heads import build,forward,ARM
    from .composite_head import CompositeHead
    from native_visible_response_bridge_v1.bridge import load_model,require_visible
    from cx_tta.stacking_adapter import read_checkpoint as old_read
    torch.set_num_threads(1);before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        gate,aux,_,_=build();composite=CompositeHead(gate,aux);response,q=load_model(plan()['current_TTA_qualification'])
    from task_adapters.adapter import read_checkpoint as gate_read
    with torch.random.fork_rng(devices=[]):
        old_meta=old_read(Path(plan()['plain_training'])/'adapter_seed0.pt')['metadata'];new_meta=gate_read(GATE_TRAINING/'adapter_seed0.pt')['metadata']
    for k in('projection_sha256','robot_config_sha256'):assert old_meta[k]==new_meta[k],k
    root=mapped(read(T/'outputs/native_eval_throughput_v3/branch10/execution_state.json')['root'])
    files=sorted(root.glob('cases/*/M_event_full/policy[0-9][0-9][0-9].npz'));assert len(files)==85
    for file in files:
        with np.load(file,allow_pickle=False)as z:legal={k.removeprefix('legal_'):z[k].copy()for k in z.files if k.startswith('legal_')};expected={k:z[k].copy()for k in('hidden','logits','raw_delta')}
        base=legal.pop('base')[0];current=forward(composite,legal,base);gate_sidecar=composite.take(base);original_gate=forward(gate,legal,base)
        for k in expected:assert np.array_equal(current[k],expected[k]),(file,k)
        for k in original_gate:assert np.array_equal(gate_sidecar[k],original_gate[k]),(file,k)
        try:composite.take(base)
        except ValueError:pass
        else:raise AssertionError('Stale gate readout reused')
    assert torch.equal(aux.action_head[2].weight[ARM],response.arm_weight);require_visible(response,q)
    import native_arm_phase_feedback_v1.episode as original
    from . import gated_episode
    from native_q0_policy_overlap_v1.episode import make_execute
    transformed=make_execute(gated_episode.execute)
    assert transformed._q0_original is gated_episode.execute
    c=Path(__file__).parent;s=(c/'gated_episode.py').read_text()
    for old,new in reversed(read(c/'gated_episode_edits.json')):assert s.count(new)==1;s=s.replace(new,old)
    assert s==Path(original.__file__).read_text()
    from .paired_actor import AuditedActor
    from native_trained_runtime_v1.actor import AuditedActor as OldActor
    from native_policy_execution_v1.actor import ActorQuery
    assert issubclass(AuditedActor,OldActor)and AuditedActor.query.__code__.co_names.count('super')==1
    assert OldActor.__mro__[1]is ActorQuery
    assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
    value=dict(passed=True,real_cached_queries=len(files),original_auxiliary_hidden_logits_raw_byte_exact=True,original_gate_readout_hidden_logits_raw_byte_exact=True,projection_and_robot_config_SHA_identical=True,original_qualified_response_ARM_weight_and_projection_unchanged=True,original_nested_GPU_owner_actor_CodeTypes_preserved=True,only_enumerated_combination_episode_source_changes=True,original_q0_policy_semantic_overlap_transform_accepts_actual_episode=True,no_extra_VLA_or_noise_queries=True,stale_gate_readout_rejected=True,CPU_RNG_preserved=True,CUDA_initialized=False,native_GPU_qualification_still_required=True,source_sha256={str(f):sha(f)for f in c.glob('*')if f.is_file()},input_sha256={str(f):sha(f)for f in files})
    f=O/'CPU_composite_actor_v2.json';assert not f.exists();f.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items()if k not in('source_sha256','input_sha256')}))

if __name__=='__main__':main()
