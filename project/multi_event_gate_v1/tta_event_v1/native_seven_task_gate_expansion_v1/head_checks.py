"""Actual original checkpoint/head/RNG and zero-increment composition checks."""
import json
import numpy as np
from .contract import T,O,read,mapped,sha,plan
from .paired_heads import build,forward,combine,ARM

def main():
    import torch
    from native_visible_response_bridge_v1.bridge import load_model,require_visible
    torch.set_num_threads(1);before=torch.random.get_rng_state().clone()
    p=plan()
    with torch.random.fork_rng(devices=[]):
        gate,aux,robots,auxrobots=build();response,q=load_model(p['current_TTA_qualification'])
    assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
    assert torch.equal(aux.action_head[2].weight[ARM],response.arm_weight)
    require_visible(response,q)
    b=T/'outputs/native_eval_throughput_v3/branch10';root=mapped(read(b/'execution_state.json')['root'])
    files=sorted(root.glob('cases/*/M_event_full/policy000.npz'));assert len(files)==10
    count=0
    for f in files:
        with np.load(f,allow_pickle=False)as z:
            legal={k.removeprefix('legal_'):z[k].copy()for k in z.files if k.startswith('legal_')}
            base=legal.pop('base')[0];saved={k:z[k].copy()for k in('hidden','logits','raw_delta')}
        old=forward(aux,legal,base);new=forward(gate,legal,base)
        for k in saved:assert np.array_equal(old[k],saved[k]),(f,k)
        with torch.no_grad():zero=response.direction(torch.from_numpy(old['hidden'][None]),torch.from_numpy(old['logits'][:,ARM][None]),torch.from_numpy(old['raw_delta'][:,ARM][None]),response.zero_fast()).numpy()[0]
        value,report=combine(new['raw_delta'],old['raw_delta'],zero)
        assert np.array_equal(value,new['raw_delta'][:,ARM])and not report['changed']
        # No hidden-coordinate transplant: the actual loaded original response
        # still verifies against its unchanged independently trained manifest.
        require_visible(response,q)
        count+=1
    assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
    try:combine(new['raw_delta'],old['raw_delta'],old['raw_delta'][:,ARM]+np.float32(.021))
    except ValueError:pass
    else:raise AssertionError('Out-of-trust transfer was accepted')
    out=dict(passed=True,actual_cached_query_inputs=count,gate_step250_head_loaded_via_original_candidate_verifier=True,original_TTA_auxiliary_head_hidden_logits_raw_bytes_exact=True,original_qualified_ARM_weight_and_projection_preserved=True,zero_fast_combination_equals_gate_candidate_byte_exact=True,CPU_RNG_preserved=True,CUDA_initialized=False,no_extra_base_sampling=True,no_claim_of_native_gate_combo_validation=True,qualification_sha256=sha(p['current_TTA_qualification']),input_sha256={str(f):sha(f)for f in files},source_sha256={str(f):sha(f)for f in (T/'native_seven_task_gate_expansion_v1').glob('*.py')})
    file=O/'CPU_paired_heads_v1.json';assert not file.exists();file.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in('source_sha256','input_sha256')}))

if __name__=='__main__':main()
