"""All prior histories independently rebuilt from actual RGB prefix ending now."""
from pathlib import Path
import sys,numpy as np
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_local_response_v2'
sys.path.insert(0,str(T/'native_arrived_feedback_v1'))
import local_data as d
from review import read,load,sha,write
model=d.module('actual_native_local_response_v2_model',C/'model.py')
assert Path(model.__file__).resolve()==(C/'model.py').resolve()
def run():
 p=read(O/'features_protocol.json');f=load(O/'features.npz');x=f['context'];previous=f['previous_row'];a=load(Path(p['local_source'])/'data.npz');meta=p['metadata']
 assert sha(Path(p['local_source'])/'data.npz')==p['local_data_sha256'];checks=1;rebuilt=0;examples=[]
 spec=read(T/'outputs/native_arrived_mask_v3/protocol.json');root=T/'outputs/native_arrived_mask_v3/collections/79fa25d4427ec9cd476539a2';work=read(root/'GPU_work_complete.json')
 for job,record in zip(spec['jobs'],work['rows']):
  ids=[i for i,m in enumerate(meta)if m['stream_index']==job['index']and previous[i]>=0]
  if not ids:continue
  data=d.LoadedStream(job);queries=list(data.queries());native=list(data.native());offset=len(queries)-1
  images=np.stack([queries[-1][0]]+[v[0][0]for v in native]);z=load(record['masks_file']);raw=z['masks'][offset:];presence=z['presence'][offset:]
  proto,known,_=d.surface.prototype(data.history['cam0'][0],data.seed_masks());refined=[];valid=[]
  for image,mask,pres in zip(images,raw,presence):
   m,v,_=d.surface.refine(image,mask,pres,proto,known,job['instruction']);refined.append(m);valid.append(v)
  refined=np.stack(refined);valid=np.stack(valid);cache={}
  for i in ids:
   r=meta[i];j=int(previous[i]);past=meta[j];left=r['start_slot'];obj=r['object_index']
   assert past['end_slot']==left and past['start_slot']==left-1 and past['end_tick']==r['start_tick'];checks+=1
   if left not in cache:
    pp,good,color=d.flow.corners(images[left-1],refined[left-1],valid[left-1])
    end,alive,fb=d.flow.track(images[:left+1],refined[:left+1],valid[:left+1],left-1,left,pp,good,color)
    cache[left]=(pp,end,alive)
   pp,end,alive=cache[left]
   assert d.old.supported(pp,alive,obj)and d.old.supported(end,alive,obj);checks+=1
   y=np.median(end[obj,alive[obj]]-pp[obj,alive[obj]],axis=0)/320
   assert np.array_equal(y,a['target'][j]);checks+=1
   dt=past['dt_s'];expected=y.astype(float)/dt
   assert np.array_equal(x[i,34:36],expected)and x[i,43]==1.;checks+=1;rebuilt+=1
   if len(examples)<4:examples.append(dict(row=i,previous=j,current_start=r['start_tick'],last_allowed_RGB_slot=left))
  print('INDEPENDENT_ARRIVED_HISTORY_RGB_PREFIX',job['index'],len(ids),flush=True)
 assert rebuilt==int((previous>=0).sum());checks+=1
 # Real current native observations with deliberately null arm/grip commands:
 n=32;xx=x[:n].copy();xx[:,34:44]=0;u=np.zeros((n,197));u[:,38]=a['command'][:n,38];u[:,39]=7.
 u[:,[32,33,36,37]]=np.clip(xx[:,[6,13,6,13]]*np.pi,0,1)/.1
 u[:,-1]=1.
 for mode in('correct','without_locality','without_history','history_only'):
  _,v,kind,gate=model.coordinates(xx,u,a['start_points'][:n],mode)
  assert np.array_equal(v,np.zeros_like(v));checks+=1
 # Preserve measured free dynamics when robot commands are null.
 xx[:,34:36]=np.array([.01,-.02]);xx[:,43]=1.
 _,v,kind,gate=model.coordinates(xx,u,a['start_points'][:n],'correct')
 assert np.array_equal(v[:,:192],np.zeros((n,192)));checks+=1
 assert np.array_equal(v[:,192:],xx[:,34:36]*(u[:,38]*20)[:,None]);checks+=1
 assert np.all((gate>=0)&(gate<=1));checks+=1
 result=dict(passed=True,checks=checks,real_previous_windows_rebuilt=rebuilt,examples=examples,
  actual_RGB_reader_cropped_at_current_start=True,current_or_future_window_target_never_used_in_history=True,
  pure_time_age_event_has_exact_zero_response_coordinates_without_motion_or_past_flow=True,
  observed_past_object_velocity_retained_as_free_dynamics=True,
  no_contact_semantic_accuracy_or_task_benefit_claim=True,source_sha256={str(p):sha(p)for p in(C/'checks.py',C/'model.py',C/'features.py')},
  input_sha256={str(O/'features_protocol.json'):sha(O/'features_protocol.json'),str(O/'features.npz'):sha(O/'features.npz')},online_updates=0,TTA_utility_proven=False)
 write(O/'checks.json',result);print('OBJECT_LOCAL_SUPPORT_AND_ACTUAL_PREFIX_CHECKS_PASS',checks,rebuilt,flush=True)
if __name__=='__main__':run()
