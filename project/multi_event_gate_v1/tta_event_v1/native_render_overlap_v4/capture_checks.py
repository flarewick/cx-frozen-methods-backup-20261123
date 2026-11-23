"""Actual original service: graph initialization waits for the renderer."""
import threading,time
import torch
from concurrent.futures import Future
from concurrent_rng import Lane
from native_expert_owner_dispatch_v1.cpu_checks import Coordinator
from native_expert_request_priority_v1.fairness import install
from native_policy_execution_v1.common import T,write,sha
from .scheduler import NativeOverlap

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 c=Coordinator();c.rng.install();rendering=threading.Event();seen=[]
 class Native:
  def render_step(self,unused):rendering.set();time.sleep(.04);rendering.clear();return 17
  take_pictures=render_step
 def operation():
  assert rendering.is_set();bridge.await_renderer();assert not rendering.is_set();seen.append('renderer_done_before_capture');return 19
 try:
  install(c,T/'outputs/native_render_overlap_v4/CPU_capture_fairness.jsonl')
  bridge=NativeOverlap(c,torch,Native(),lambda:True,CPU_fixture_codes={operation.__code__},on_progress=lambda v:seen.append(v))
  lane=Lane('model');future=Future();c.service.requests.put((0,0,lane,operation,(),{},future))
  bridge.last_progress-=6
  def observed():
   with c.scene_gate.read(),bridge.active(True):return bridge.render_step(None)
  assert c.call(Lane('physical'),observed)==17 and future.result()==19
  assert seen[0]=='renderer_done_before_capture'and seen[1]['bookkeeping_only']and not seen[1]['worker_pool_closed']and not seen[1]['passed']
  bridge.close();assert bridge.closed and not c.scene_gate.readers
  write(T/'outputs/native_render_overlap_v4/CPU_capture_wait_checks_v1.json',dict(passed=True,CPU_only=True,actual_original_service_renderer_exclusion_before_capture=True,partial_progress_not_scientific_or_success_proof=True,original_worker_released=True,no_CUDA_or_scientific_qualification=True))
  print('ORIGINAL_RENDERER_EXCLUSION_AND_PARTIAL_PROGRESS_CPU_PASS',flush=True)
 finally:c.rng.uninstall()
if __name__=='__main__':main()
