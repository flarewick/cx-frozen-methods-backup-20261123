"""Exact same-device original G reference; archived G differences remain explicit diagnostics."""
import numpy as np
from .common import C,read,write,sha
from .official_actor import AuditedActor as Original
class AuditedActor(Original):
 def __init__(self,*args,**kwargs):
  c=args[0];native=c.rng.current();state=c.call(native,lambda:c.rng.capture(native));super().__init__(*args,**kwargs);c.call(self.sampler,lambda:c.rng.restore(self.sampler,state))
  self.case_id=kwargs['name']
  for suffix in ('_M_event_full','_zero_update','_shadow_update','_TTA'):
   if self.case_id.endswith(suffix):self.case_id=self.case_id[:-len(suffix)];break
  self.golden=read(C/'pi05_gt_shared6_v1/original_G_query0_manifest_v1.json')['records'][self.case_id]
 def query(self,observed,record,folder):
  reference=None
  if record['query']==0:
   def actual_original():
    from .original_reference import predict
    before=self.c.rng.capture(self.sampler)
    try:return predict(self.base,self.head,self.robots,self.projection,observed,self.instruction,int(self.case_id.rsplit(':',1)[1]))
    finally:self.c.rng.restore(self.sampler,before)
   reference=self.c.call(self.sampler,actual_original)
  result,row,values=super().query(observed,record,folder)
  if reference is not None:
   assert values['base'].tobytes()==reference['base'].tobytes(),'Same-device original G base mismatch; no policy actions allowed'
   assert values['raw_delta'].tobytes()==reference['raw_delta'].tobytes(),'Same-device original G head mismatch; no policy actions allowed'
   for key,value in reference['legal'].items():assert np.array_equal(values['legal_'+key],value),key
   old=self.golden;assert sha(old['query0_file'])==old['query0_sha256']
   with np.load(old['query0_file'],allow_pickle=False)as z:archived=z['base'].copy()
   write(folder/'query000_original_G_match.json',dict(passed=True,base_byte_exact=True,head_byte_exact=True,all_legal_inputs_byte_exact=True,reference_kind='unchanged_original_G_predict_body_same_process_device',original_reference_query_discarded_before_actions=True,RNG_restored_after_reference=True,official_native_scene_rng_bound=True,no_policy_actions_before_this_check=True,historical_base_byte_exact=np.array_equal(values['base'],archived),historical_base_max_abs=float(np.max(np.abs(values['base']-archived))),historical_G_not_substituted_for_new_matched_G=True,reference=old))
  return result,row,values
