"""Whole-cache actual VLA reference/witness equality; no simulator."""
import time
from pathlib import Path
import numpy as np
import torch
from concurrent_rng import RngRouter,Lane
from sampler_fork import audited_draw,full_hash,environment_hash
from query_ledger import QueryLedger
from query_current import query_current
from sampler_gpu_spec import read,write,sha,verify,C,T,O
from native_stage_v1.cx_tta_stage.query import build_head
from native_stage_v1.cx_tta_stage.views import view_inputs,batched_encode
from cx_tta.structured_features import project
from cx_tta.eval_diagnostics import array_hash
def seed(router,lane,n):
 import random
 with router.bind(lane),router.torch_scope(lane):np.random.seed(n);random.seed(n);torch.manual_seed(n)
def execute(policy,spec,root,barrier,device_uuid,jobs):
 assert torch.cuda.device_count()==1;torch.set_num_threads(2)
 from adapter_cache_cpu import move_to_gpu
 base=move_to_gpu(policy);base._model.eval().requires_grad_(False);sampler=base._model.sample_actions
 while hasattr(sampler,'_torchdynamo_orig_callable'):sampler=sampler._torchdynamo_orig_callable
 base._sample_actions=sampler;torch.cuda.synchronize()
 write(barrier/'cuda_ready.json',dict(inner_pid=__import__('os').getpid(),gpu_uuid=device_uuid))
 began=time.monotonic()
 while not (barrier/'owned_pid_permit.json').exists():
  if time.monotonic()-began>45:raise TimeoutError('No live verified GPU PID permit')
  time.sleep(.03)
 assert read(barrier/'owned_pid_permit.json')['gpu_uuid']==device_uuid
 router=RngRouter(cuda=True);router.install();head,bots=build_head(spec['training'])
 with np.load(spec['projection'],allow_pickle=False) as z:projection={k:z[k].copy() for k in z.files}
 reports=[];total=0;start=time.monotonic()
 try:
  for index,(job,data) in enumerate(zip(spec['jobs'],jobs)):
   states=data['states'];n=len(states);features=[];encoder_lane=Lane('encode'+str(index));seed(router,encoder_lane,spec['seed'])
   with router.bind(encoder_lane),router.torch_scope(encoder_lane):
    for q in range(n):
     images=tuple(data['cam'+str(c)][q] for c in range(3));anchor=tuple(data['cam'+str(c)][0] for c in range(3))
     before=full_hash(router.capture(encoder_lane));joint,meta=batched_encode(base,view_inputs(images,states[q],anchor,states[0]),job['instruction'])
     assert full_hash(router.capture(encoder_lane))==before
     visual,_=project(joint,projection)
     features.append(dict(joint=joint,visual=visual,command_state=states[q].copy(),rgb=images,images={str(c):array_hash(im) for c,im in enumerate(images)},observation_tick=int(data['query_ticks'][q])))
   results={};notes={};case=root/('case%02d'%index);case.mkdir()
   for mode in spec['passes']:
    lane=Lane(mode+str(index));seed(router,lane,spec['seed']);bases=[];deltas=[];legals=[];trace=[];previous=None
    with router.bind(lane),router.torch_scope(lane):
     for q,current in enumerate(features):
      tick=current['observation_tick'];op=lambda:query_current(base,head,bots,projection,current,job['instruction'],previous,decision_tick=tick);before=router.capture(lane)
      if mode=='unmodified_reference':
       result=op();after=router.capture(lane);assert environment_hash(before)==environment_hash(after)
       note=dict(query=q,tick=tick,before=full_hash(before),after=full_hash(after),actor_images=result['actor_images'])
      else:
       # Separate one-draw cached contexts. Do not fabricate executed32 receipts.
       ledger=QueryLedger('cached_observation_'+str(q))
       result,after,witness=audited_draw(router,lane,base._model,ledger,op,query=0,tick=tick)
       note=dict(query=q,tick=tick,before=full_hash(before),after=full_hash(after),actor_images=result['actor_images'],noise_witness=witness,cached_context_not_executed=True)
      bases.append(result['base']);deltas.append(result['raw_delta']);legals.append(result['legal']);trace.append(note);previous=result['previous']
    arrays=dict(base=np.stack(bases),raw_delta=np.stack(deltas),**{'legal_'+k:np.concatenate([r[k] for r in legals],axis=0) for k in legals[0]})
    path=case/(mode+'.npz')
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    write(case/(mode+'.json'),dict(queries=n,trace=trace,file_sha256=sha(path)));results[mode]=arrays;notes[mode]=trace
   a,b=[results[k] for k in spec['passes']]
   for key in a:assert a[key].tobytes()==b[key].tobytes(),(index,key)
   at,bt=[notes[k] for k in spec['passes']]
   for left,right in zip(at,bt):
    for key in ('query','tick','before','after','actor_images'):assert left[key]==right[key],(index,key)
   total+=n;reports.append(dict(case_id=job['case_id'],queries=n,all_base_delta_legal_arrays_and_RNG_exact=True))
   progress=root/'progress.json';temp=progress.with_suffix('.json.tmp');temp.write_text(__import__('json').dumps(dict(completed_cases=index+1,queries=total,seconds=time.monotonic()-start,simulator_started=False))+'\n');temp.replace(progress)
   print('REAL_VLA_SAMPLER_CASE',index,n,total,round(time.monotonic()-start,3),flush=True)
  verify(spec)
  write(root/'report.json',dict(complete=True,cases=len(reports),cached_queries=total,base_infer_calls=total*2,cases_report=reports,protocol_identity=spec['identity'],gpu_uuid=device_uuid,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start,scope=spec['scope'],actual_full_VLA_weights_used=True,
    source_sampler_infer_no_external_noise=True,all_ref_witness_base_delta_legal_RNG_exact=True,simulator_started=False,online_updates=0,TTA_utility_proven=False))
  print('REAL_VLA_SAMPLER_WITNESS_COMPLETE',total,flush=True)
 finally:router.uninstall()
