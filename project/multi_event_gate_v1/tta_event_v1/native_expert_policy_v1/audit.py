"""Independent CPU reconstruction of actual frozen-head outputs and zero updates."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from native_expert_policy_v1.paths import read,write,sha,verify,O
from cx_tta.structured_action_adapter import FORWARD_KEYS,ARM_INDICES
from cx_tta.structured_features import pack_observable
from cx_tta.eval_diagnostics import array_hash
from native_nextchunk_v1.sampler_fork import base_hash

def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy() for k in z.files}

def audit(spec,root,head=None,robots=None,projection=None):
    assert not torch.cuda.is_initialized();torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    before=torch.random.get_rng_state().clone()
    if head is None:
        from native_feedback_confirm_v5.cx_tta_feedback_confirm5.query import build_head
        with torch.random.fork_rng(devices=[]):head,robots=build_head(spec['training'])
    if projection is None:projection=load(spec['projection'])
    original={k:v.detach().clone() for k,v in head.state_dict().items()}
    data=read(root/'GPU_cache_complete.json');manifest=read(spec['query_manifest'])
    assert data['complete'] and data['protocol_identity']==spec['identity']
    assert data['base_infer_calls']==data['actual_internal_noise_calls']==data['queries']==36
    assert len(data['rows'])==len(manifest['records'])==36
    captured={}
    def hook(name):
        def observe(module,args,value):captured[name]=value.detach().cpu().numpy().copy()
        return observe
    hooks=[head.action_head[1].register_forward_hook(hook('hidden')),head.action_head[2].register_forward_hook(hook('logits'))]
    checks=0;prior={};arm=list(ARM_INDICES);output=[]
    try:
        for row,reference in zip(data['rows'],manifest['records'],strict=True):
            assert row['case_id']==reference['case_id'] and row['query']==reference['query']
            assert row['actor_observation_tick']==reference['observation_global_tick'];checks+=2
            assert row['observation_file']==reference['observation_file']
            assert sha(row['observation_file'])==row['observation_sha256']==reference['observation_sha256'];checks+=2
            assert sha(row['policy_file'])==row['policy_sha256'];a=load(row['policy_file']);obs=load(row['observation_file']);checks+=1
            assert row['actor_images']=={str(c):array_hash(obs['cam'+str(c)]) for c in range(3)}
            assert row['command_state_sha256']==array_hash(obs['command_state']);checks+=2
            assert row['base_infer_calls']==row['actual_internal_noise_calls']==1 and not row['sampled_base_executed'];checks+=1
            w=row['noise_witness'];assert w['base_sha256']==base_hash(a['base']) and w['tick']==row['actor_observation_tick']
            assert w['environment_unchanged'] and w['before']!=w['after'] and w['rng_before']!=w['rng_after'] and w['shape']==[1,32,32];checks+=2
            key=row['case_id'];q=row['query'];joint=a['joint'][0].copy();state=obs['command_state']
            if q==0:assert key not in prior;pj=np.zeros_like(joint);ps=np.zeros_like(state)
            else:
                pq,pj,ps=prior[key];assert pq==q-1
            expected=pack_observable(joint[None],state[None],pj[None],ps[None],np.asarray([q>0]),projection,robots)
            prior[key]=(q,joint,state.copy())
            legal={k.removeprefix('legal_'):v for k,v in a.items() if k.startswith('legal_') and k!='legal_base'}
            assert set(legal)==set(expected)==FORWARD_KEYS;checks+=1
            for k in expected:
                assert np.array_equal(legal[k],expected[k]),(key,q,k);checks+=1
            assert np.array_equal(a['legal_base'],a['base'][None]);checks+=1;captured.clear()
            with torch.inference_mode():
                result=head({k:torch.from_numpy(v.copy()) for k,v in legal.items()},torch.from_numpy(a['base'][None].copy()),enabled=True)
            assert np.array_equal(result['delta'][0].numpy(),a['raw_delta']);checks+=1
            assert np.array_equal(captured['hidden'][0],a['hidden']) and np.array_equal(captured['logits'][0],a['logits']);checks+=2
            h=torch.from_numpy(a['hidden'][None].copy());logits=torch.from_numpy(a['logits'][None].copy())
            assert h.shape==(1,32,64) and logits.shape==(1,32,14)
            assert torch.equal(F.linear(h,head.action_head[2].weight,head.action_head[2].bias),logits);checks+=2
            ref=torch.from_numpy(a['raw_delta'][None].copy())[...,arm];z=torch.zeros(64)
            inc=F.linear(h*z+z,head.action_head[2].weight[arm]);l=logits[...,arm]
            reconstructed=(ref+.1*(torch.tanh(l+inc)-torch.tanh(l+torch.zeros_like(inc)))).clamp(-.1,.1)
            assert torch.equal(reconstructed,ref) and not np.any(a['raw_delta'][:,[6,13]]);checks+=2
            output.append(dict(case_id=key,query=q,actor_observation_tick=row['actor_observation_tick'],policy_file=row['policy_file'],policy_sha256=row['policy_sha256'],zero_fast128_exact=True))
    finally:
        for h in hooks:h.remove()
    assert len(prior)==4 and torch.equal(before,torch.random.get_rng_state())
    assert all(torch.equal(original[k],v) for k,v in head.state_dict().items()) and not torch.cuda.is_initialized();checks+=2
    report=dict(passed=True,checks=checks,queries=len(output),rows=output,
        actual_once_base_hidden64_logits_exact=True,legal_previous_history_reconstructed=True,zero_fast128_exact=True,
        actual_future_expert_targets_NOT_read=True,post_success_frames_queried=0,no_optimizer_or_training=True,
        sampled_base_NOT_executed=True,causal_feedback_still_missing=True,native_training_cache_ready=False,
        online_updates=0,TTA_utility_proven=False)
    f=root/'CPU_qualification.json'
    if f.exists():assert read(f)==report
    else:write(f,report)
    print('EXPERT_POLICY_CACHE_CPU_PASS',len(output),checks,flush=True);return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    spec=read(O/'protocol.json');verify(spec);audit(spec,a.dataset)
