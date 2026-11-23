"""Add only the exact preceding arrived correspondence window to original context53."""
from pathlib import Path
import json,sys,numpy as np
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_local_response_v2'
sys.path.insert(0,str(T/'native_arrived_feedback_v1'))
from review import read,load,sha,write
def run():
 original=T/'outputs/native_origin_response_v1';p=read(original/'features_protocol.json');a=load(Path(p['local_source'])/'data.npz')
 assert read(original/'study/independent_audit.json')['passed']
 x=load(original/'features.npz')['context'].copy();assert sha(original/'features.npz')==p['features_sha256'];meta=p['metadata']
 byend={}
 for i,r in enumerate(meta):
  key=(r['case_id'],r['branch'],r['object_index'],r['end_tick'])
  assert key not in byend;byend[key]=i
 previous=np.full(len(meta),-1,np.int64);checks=0;cross_stage=0
 for i,r in enumerate(meta):
  j=byend.get((r['case_id'],r['branch'],r['object_index'],r['start_tick']))
  if j is None:continue
  past=meta[j];assert past['start_tick']<past['end_tick']==r['start_tick'];dt=past['dt_s'];assert dt>0
  # All values are observations/commands from the preceding window, ending now.
  grip0=a['context'][j,[6,13]]*np.pi;grip1=a['command'][j,36:38]*.1
  x[i,34:44]=np.r_[a['target'][j].astype(float)/dt,a['command'][j,12:16]/dt,(grip1-grip0)/dt,dt,1.]
  previous[i]=j;checks+=1
  if past['segment']!=r['segment']:cross_stage+=1
 assert np.array_equal(x[:,:34],load(original/'features.npz')['context'][:,:34]);checks+=1
 assert np.array_equal(x[:,44:],load(original/'features.npz')['context'][:,44:]);checks+=1
 assert np.isfinite(x).all()and np.all(x[previous<0,34:44]==0);checks+=1
 O.mkdir(parents=True,exist_ok=True)
 with(O/'features.npz').open('xb')as f:np.savez_compressed(f,context=x,previous_row=previous)
 report=dict(complete=True,rows=len(meta),checks=checks,past_observed_rows=int((previous>=0).sum()),past_origin_to_following_rows=cross_stage,
  metadata=meta,local_source=p['local_source'],local_data_sha256=p['local_data_sha256'],missing_cases=p['missing_cases'],
  features_sha256=sha(O/'features.npz'),original_features_protocol_sha256=sha(original/'features_protocol.json'),
  source_sha256={str(C/'features.py'):sha(C/'features.py')},
  history_measurement_source='Only previous native RGB correspondence ending exactly at current start; not current/future target. Previous full-stage update receipt is not used as an earlier policy update.',
  needs_independent_raw_RGB_history_check=True,targets_and_command197_unchanged=True,online_updates=0,TTA_utility_proven=False)
 write(O/'features_protocol.json',report);print('STRICT_PREVIOUS_NATIVE_WINDOW_CONTEXT_COMPLETE',len(meta),int((previous>=0).sum()),cross_stage,flush=True)
if __name__=='__main__':run()
