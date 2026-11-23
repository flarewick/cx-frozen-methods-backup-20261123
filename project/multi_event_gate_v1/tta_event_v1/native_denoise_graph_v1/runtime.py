"""Fresh paired RNG forks; compare entire original sampler actions, no environment."""
import time,pickle,random,copy
from pathlib import Path
from .common import read,write,verify,sha

def execute(spec,root,prepared):
 import torch,numpy as np
 from adapter_cache_cpu import move_to_gpu
 from concurrent_rng import RngRouter,Lane
 from native_nextchunk_v1.sampler_fork import audited_draw,full_hash
 from native_nextchunk_v1.query_ledger import QueryLedger
 from .graph import DenoiseGraph
 base=move_to_gpu(prepared['base']);model=base._model;model.eval().requires_grad_(False);torch.set_num_threads(2)
 sampler=model.sample_actions
 while hasattr(sampler,'_torchdynamo_orig_callable'):sampler=sampler._torchdynamo_orig_callable
 base._sample_actions=sampler;graph=DenoiseGraph(model);original=model.denoise_step
 versions=[(n,p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for n,p in model.named_parameters()]
 write(root/'barrier/cuda_ready.json',dict(ready=True));began=time.monotonic()
 while not(root/'barrier/owned_pid_permit.json').exists():
  if time.monotonic()-began>45:raise TimeoutError('No verifiedGPUownerpermit')
  time.sleep(.03)
 router=RngRouter(cuda=True);lane=Lane('fresh_software_sampler');router.install();rows=[]
 try:
  with router.bind(lane),router.torch_scope(lane):
   random.seed(17);np.random.seed(17);torch.manual_seed(17)
   for item in spec['probe_inputs']:
    f=Path(item['file']);assert sha(f)==item['sha256']
    with np.load(f,allow_pickle=False)as z:images=[z['cam'+str(i)].copy()for i in range(3)];state=z['command_state'].copy()
    obs=dict(state=state,prompt=item['instruction'],images={k:im.transpose(2,0,1).copy()for k,im in zip(('cam_high','cam_right_wrist','cam_left_wrist'),images)})
    before=router.capture(lane);model.denoise_step=original;torch.cuda.synchronize();t=time.perf_counter()
    a,sa,wa=audited_draw(router,lane,model,QueryLedger('original_software'),lambda:dict(base=np.asarray(base.infer(copy.deepcopy(obs))['actions'],np.float32)),query=0,tick=0);torch.cuda.synchronize();old=time.perf_counter()-t
    router.restore(lane,before);model.denoise_step=graph;t=time.perf_counter()
    b,sb,wb=audited_draw(router,lane,model,QueryLedger('graph_software'),lambda:dict(base=np.asarray(base.infer(copy.deepcopy(obs))['actions'],np.float32)),query=0,tick=0);torch.cuda.synchronize();new=time.perf_counter()-t
    exact=a['base'].dtype==b['base'].dtype and a['base'].shape==b['base'].shape and a['base'].tobytes()==b['base'].tobytes();rng=full_hash(sa)==full_hash(sb);noise=wa==wb
    row=dict(file=str(f),action_bytes_exact=exact,RNG_exact=rng,once_noise_witness_exact=noise,original_seconds=old,graph_seconds=new,includes_capture=len(rows)==0,max_abs_difference=float(np.max(np.abs(a['base']-b['base']))),graph_denoise_calls=graph.calls)
    rows.append(row);write(root/('case_%02d.json'%(len(rows)-1)),row);print('DENOISE_GRAPH_ACTUAL',row,flush=True)
    assert exact and rng and noise,'Denoise graph changed original sampler actions/noise/RNG'
 finally:model.denoise_step=original;router.uninstall()
 assert versions==[(n,p.data_ptr(),p._version,str(p.dtype),tuple(p.shape))for n,p in model.named_parameters()];verify(spec);old=float(np.median([r['original_seconds']for r in rows[1:]]));new=float(np.median([r['graph_seconds']for r in rows[1:]]))
 write(root/'GPU_probe_result.json',dict(passed=True,actual_cases=len(rows),all_final_actions_noise_RNG_byte_exact=True,model_parameter_pointers_versions_unchanged=True,original_outer_sampler_and_noise_unchanged=True,software_infer_calls=2*len(rows),policy_episodes=0,policy_actions=0,original_median_seconds=old,graph_steady_median_seconds=new,steady_speedup=old/new,capture_seconds=rows[0]['graph_seconds'],rows=rows,sustained_resource_qualification_NOT_claimed=True))
