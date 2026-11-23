"""First semantically available retained prefix/task; never inspect task outcomes."""
import numpy as np
from .common import T,O,read,write,sha,mapped
TASKS=('beat_block_hammer','turn_switch','adjust_bottle','grab_roller','open_laptop')
def save(dest,values):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  with np.load(dest,allow_pickle=False)as z:
   assert set(z.files)==set(values)and all(z[k].dtype==v.dtype and z[k].shape==v.shape and z[k].tobytes()==v.tobytes()for k,v in values.items())
 else:
  with dest.open('xb')as output:np.savez_compressed(output,**values)
def prepare():
 base=T/'outputs/native_onpolicy_arm_collection_v17/val_000';root=mapped(read(base/'execution_state.json')['root'])
 rows=[];bindings={};counts=set()
 for task in TASKS:
  for folder in sorted((root/'cases').glob(task+':*'),key=lambda p:int(p.name.split(':')[1])):
   f=folder/'M_event_full';receipt=f/'semantic_initial_receipt.json';dense=f/'execution000_arrivals.npz'
   if not receipt.exists()or not dense.exists():continue
   r=read(receipt)
   if not r['available']:continue
   answer=mapped(r['CPU_result_file']);assert sha(answer)==r['CPU_result_sha256'];a=read(answer)
   masks=mapped(a['mask_file']);assert sha(masks)==a['mask_sha256']
   initial=f/'observation000.npz'
   with np.load(initial,allow_pickle=False)as z:rgb=z['cam0'].copy()
   with np.load(mapped(a['frame_file']),allow_pickle=False)as z:assert np.array_equal(rgb,z['cam0'])
   with np.load(masks,allow_pickle=False)as z:mask=z['masks'].copy()
   with np.load(dense,allow_pickle=False)as z:
    index=np.flatnonzero(z['ticks']>0)[:8];assert len(index)==8
    frames=z['head_rgb'][index].copy();ticks=z['ticks'][index].copy()
   count=2 if mask[1].any()else 1;assert mask[0].any();counts.add(count)
   dest=O/'inputs'/(task+'.npz');save(dest,dict(initial=rgb,masks=mask,frames=frames,ticks=ticks))
   rows.append(dict(task=task,case_id=folder.name,file=str(dest),actual_object_count=count,actual_native_arrivals=8))
   bindings.update({str(p):sha(p)for p in [receipt,answer,masks,initial,dense,mapped(a['frame_file']),dest]});break
  else:raise AssertionError(('Missing real available first8 prefix',task))
 # New tasks here all have one real object. Add one existing TRAIN developer
 # blocks prefix for actual two-object compatibility, never EVAL or fitting.
 original=T/'outputs/native_selected_runtime_v2';protocol=read(original/'protocol.json')
 assert protocol['scenes'][0]['case_id']=='stack_blocks_two:7100000'
 folder=original/'collections/d992551309005b55cf58d596/case000/shadow_update'
 seed=T/'outputs/native_trained_semantics_v1/case000_masks.npz'
 initial=T/'outputs/native_trained_initial_v2/collections/826b03c50966578d54a6cb97/case000/initial.npz'
 with np.load(seed,allow_pickle=False)as z:mask=z['masks'].copy()
 with np.load(initial,allow_pickle=False)as z:rgb=z['cam0'].copy()
 with np.load(folder/'observation000.npz',allow_pickle=False)as z:assert np.array_equal(rgb,z['cam0'])
 assert mask.reshape(2,-1).any(1).all();counts.add(2)
 result=read(folder/'result.json');assert len(result['queries'])>=9
 frames=[];ticks=[];files=[original/'protocol.json',seed,initial,folder/'result.json',folder/'observation000.npz']
 for query in result['queries'][1:9]:
  file=mapped(query['observation']['file']);assert sha(file)==query['observation']['sha256']
  with np.load(file,allow_pickle=False)as z:frames.append(z['cam0'].copy())
  ticks.append(query['tick']);files.append(file)
 dest=O/'inputs/blocks_TRAIN_compatibility.npz';save(dest,dict(initial=rgb,masks=mask,frames=np.stack(frames),ticks=np.asarray(ticks,np.int64)))
 rows.append(dict(task='stack_blocks_two_TRAIN_software_compatibility_only',case_id='stack_blocks_two:7100000',file=str(dest),actual_object_count=2,actual_native_arrivals=8))
 bindings.update({str(p):sha(p)for p in [*files,dest]})
 assert counts=={1,2},('Actual one/two-object coverage required',counts)
 write(O/'input_manifest.json',dict(passed=True,cases=rows,known_object_counts=sorted(counts),software_only=True,failed_source_never_scientific_data=True,no_success_or_expert_label_selection=True,input_sha256=bindings))
 return rows
