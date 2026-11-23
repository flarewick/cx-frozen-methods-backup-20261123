"""One witnessed real base per expert query; no simulation or teacher inputs."""
import os,time,random
import numpy as np
import torch
from concurrent_rng import RngRouter,Lane
from native_nextchunk_v1.sampler_fork import audited_draw,full_hash
from native_nextchunk_v1.query_ledger import QueryLedger
from native_nextchunk_v1.query_current import query_current
from native_feedback_confirm_v5.cx_tta_feedback_confirm5.views import view_inputs,batched_encode
from cx_tta.eval_diagnostics import array_hash
from native_expert_policy_v1.paths import read,write,sha

def seed(router,lane,n):
    with router.bind(lane),router.torch_scope(lane):
        random.seed(n);np.random.seed(n);torch.manual_seed(n)

def execute(spec,root,barrier,prepared):
    assert torch.cuda.device_count()==1;torch.set_num_threads(2)
    from adapter_cache_cpu import move_to_gpu
    base=move_to_gpu(prepared['policy']);base._model.eval().requires_grad_(False)
    sampler=base._model.sample_actions
    while hasattr(sampler,'_torchdynamo_orig_callable'):sampler=sampler._torchdynamo_orig_callable
    base._sample_actions=sampler;torch.cuda.synchronize()
    assert torch.get_float32_matmul_precision()=='high'
    write(barrier/'cuda_ready.json',dict(inner_pid=os.getpid(),gpu_uuid=spec['required_gpu_uuid']))
    began=time.monotonic()
    while not(barrier/'owned_pid_permit.json').exists():
        if time.monotonic()-began>45:raise TimeoutError('No verified GPU owner permit')
        time.sleep(.03)
    assert read(barrier/'owned_pid_permit.json')['gpu_uuid']==spec['required_gpu_uuid']
    head=prepared['head'];robots=prepared['robots'];projection=prepared['projection']
    original={k:v.detach().clone() for k,v in head.state_dict().items()};captured={}
    def hook(name):
        def observe(module,args,value):captured[name]=value.detach().cpu().numpy().copy()
        return observe
    hooks=[head.action_head[1].register_forward_hook(hook('hidden')),
           head.action_head[2].register_forward_hook(hook('logits'))]
    router=RngRouter(cuda=True);router.install();rows=[];start=time.monotonic()
    try:
        for index,(job,data) in enumerate(zip(spec['jobs'],prepared['jobs'],strict=True)):
            encoder=Lane('expert_encoder_'+str(index));lane=Lane('expert_base_'+str(index))
            seed(router,encoder,spec['seed']);seed(router,lane,spec['seed']);previous=None
            for query,arrays in zip(job['queries'],data,strict=True):
                q=query['query'];tick=query['tick'];images=tuple(arrays['cam'+str(c)] for c in range(3));state=arrays['command_state']
                with router.bind(encoder),router.torch_scope(encoder):
                    before=full_hash(router.capture(encoder))
                    joint,encoded=batched_encode(base,view_inputs(images,state,images,state),job['instruction'])
                    assert full_hash(router.capture(encoder))==before
                current=dict(joint=joint,command_state=state.copy(),rgb=images,
                             images={str(c):array_hash(im) for c,im in enumerate(images)},observation_tick=tick)
                # Independent one-draw cache ledger: never attach execution receipts for this unexecuted base.
                ledger=QueryLedger('offline_actual_state_'+str(q));captured.clear()
                with router.bind(lane),router.torch_scope(lane):
                    operation=lambda:query_current(base,head,robots,projection,current,job['instruction'],previous,decision_tick=tick)
                    result,after,witness=audited_draw(router,lane,base._model,ledger,operation,query=0,tick=tick)
                assert ledger.awaiting_execution and len(ledger.records)==1
                assert captured['hidden'].shape==(1,32,64) and captured['logits'].shape==(1,32,14)
                assert result['actor_images']==current['images'] and result['actor_observation_tick']==tick
                previous=result['previous'];folder=root/('record%03d'%len(rows));folder.mkdir()
                file=folder/'policy.npz'
                values=dict(base=result['base'],raw_delta=result['raw_delta'],hidden=captured['hidden'][0],
                    logits=captured['logits'][0],joint=joint,**{'legal_'+k:v for k,v in result['legal'].items()})
                assert all(np.isfinite(v).all() for v in values.values())
                with file.open('xb')as f:np.savez_compressed(f,**values)
                row=dict(case_id=job['case_id'],query=q,actor_observation_tick=tick,
                    observation_file=query['observation_file'],observation_sha256=query['observation_sha256'],
                    actor_images=result['actor_images'],command_state_sha256=array_hash(state),
                    noise_witness=witness,encoder=encoded,policy_file=str(file),policy_sha256=sha(file),
                    base_infer_calls=1,actual_internal_noise_calls=1,sampled_base_executed=False,
                    prior_query_history_only=True,actual_online_query_claimed=False)
                write(folder/'query.json',row);rows.append(row)
                print('EXPERT_REAL_POLICY_QUERY',index,q,len(rows),round(time.monotonic()-start,3),flush=True)
        assert len(rows)==36
        assert all(torch.equal(original[k],v) for k,v in head.state_dict().items())
        write(root/'GPU_cache_complete.json',dict(complete=True,protocol_identity=spec['identity'],rows=rows,
            base_infer_calls=len(rows),actual_internal_noise_calls=len(rows),queries=len(rows),
            seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),simulator_started=False,
            native_training_cache_ready=False,training_steps=0,online_updates=0,TTA_utility_proven=False))
    finally:
        for h in hooks:h.remove()
        router.uninstall()
