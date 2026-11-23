"""A failed historical run is a read-only software oracle, never training data."""
from pathlib import Path
import threading,time
import numpy as np
from native_policy_execution_v1.common import read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from .defer import TASKS

class ParityProbeComplete(RuntimeError):pass

def factories(original_gate,original_factory,root,spec):
 class Gate(original_gate):
  def release(self,reason,**kwargs):
   if reason=='actual_complete_initial_native32':return False
   raise RuntimeError('Unexpected whole terminal before diagnostic next observation')
 def factory(*args,**kwargs):
  original=original_factory(*args,**kwargs)
  class Actor(original):
   def query(self,observed,record,folder):
    case=folder.parent.name;q=record['query']
    if q==0:
     if case==spec['software_reference_case']:
      before=read(mapped(spec['software_reference_folder'])/'query000_environment_audit.json')
      assert observed['physical_sha256']==before['physical_sha256']and observed['rng_sha256']==before['rng_sha256']
      compare_npz(mapped(spec['software_reference_folder'])/'observation000.npz',mapped(record['file']))
     return super().query(observed,record,folder)
    assert q==1,'At most one actual policy query per probe scene'
    if case!=spec['software_reference_case']:
     while not args[1].cancel.wait(.05):pass
     raise InterruptedError('Peer diagnostic ended')
    reference=mapped(spec['software_reference_folder']);checks=[]
    for name in ['observation000.npz','policy000.npz','actions000.npz','execution000.npz','execution000_arrivals.npz','observation001.npz']:
     checks.append(compare_npz(reference/name,folder/name))
    old=read(reference/'policy000.json');new=read(folder/'policy000.json')
    assert old['noise_witness']==new['noise_witness'],'Original once-noise/RNG witness changed'
    expected=read(reference/'query001_environment_audit.json')
    assert observed['physical_sha256']==expected['physical_sha256']and observed['rng_sha256']==expected['rng_sha256']
    task=TASKS[case]();assert task is not None and task._cx_deferred_capture_count>0
    write(root/'engineering_render_parity.json',dict(passed=True,software_probe_only=True,scientific_qualification=False,
      case_id=case,arrays=checks,visible_RGB_actions_physics_RNG_noise_exact=True,
      skipped_unread_captures=task._cx_deferred_capture_count,actual_policy_queries_in_target=1,
      next_observation_arrived_without_sampling_query1=True,monotonic_ns=time.monotonic_ns()))
    raise ParityProbeComplete('Declared first32/next-observation software parity probe ended')
  return Actor
 return Gate,factory

def compare_npz(reference,current):
 with np.load(reference,allow_pickle=False)as a,np.load(current,allow_pickle=False)as b:
  assert set(a.files)==set(b.files),(reference.name,'array keys')
  for name in a.files:
   x,y=a[name],b[name]
   assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),(reference.name,name,'byte mismatch')
  return dict(file=reference.name,arrays=len(a.files),all_bytes_exact=True)
