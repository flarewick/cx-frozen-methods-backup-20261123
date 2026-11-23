"""Canonical qualified first8 FIT inputs; no old semantics/outputs/labels."""
from pathlib import Path
import numpy as np
from .common import T,read,sha,mapped,identity,verify
from native_visible_expand_semantics_v2.store import key
def collect(initial,tasks,limit,seed_start):
 inputs={};roots=[]
 def bind(f):f=mapped(f);inputs[str(f)]=sha(f);return read(f)
 for namespace in('native_onpolicy_full_v1','native_onpolicy_coverage_v8'):
  o=T/'outputs'/namespace/'pilot_fit';r=mapped(bind(o/'execution_state.json')['root']);spec=bind(r/'protocol.json');verify(spec);assert bind(r/'terminal_audit.json')['passed']and bind(o/'actual_host_wait_exit.json')['exit_code']==0;roots.append((r,spec))
 # Preserve earliest completed validity decisions, including negatives.
 for version in('native_onpolicy_collection_v1','native_onpolicy_collection_v2','native_onpolicy_collection_v3','native_onpolicy_collection_v7','native_onpolicy_collection_v8','native_onpolicy_collection_v9','native_onpolicy_collection_v10','native_onpolicy_collection_v11','native_onpolicy_collection_v12','native_onpolicy_collection_v13','native_onpolicy_collection_v14','native_onpolicy_collection_v15','native_onpolicy_collection_v16','native_onpolicy_arm_collection_v2','native_onpolicy_arm_collection_v3','native_onpolicy_arm_collection_v4','native_onpolicy_arm_collection_v5','native_onpolicy_arm_collection_v6','native_onpolicy_arm_collection_v9','native_onpolicy_arm_collection_v10','native_onpolicy_arm_collection_v17','native_onpolicy_arm_collection_v35'):
  base=T/'outputs'/version
  for o in sorted(base.glob('*_*')):
   if not(o/'actual_host_wait_exit.json').exists()or not(o/'execution_state.json').exists():continue
   actual=bind(o/'actual_host_wait_exit.json');r=mapped(bind(o/'execution_state.json')['root']);spec=bind(r/'protocol.json');verify(spec)
   if actual['exit_code']==0:
    assert bind(r/'atomic_full_recovery_qualification.json')['passed'];roots.append((r,spec))
    future=r/'next_preparation'
    if(future/'preparation_complete.json').exists():
     assert bind(future/'preparation_complete.json')['complete'];roots.append((future,spec))
   elif version=='native_onpolicy_arm_collection_v6'and o.name=='val_000':
    import json
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    assert bind(r/'signal_received_21.json')['signal']==2 and bind(r/'attempt/barrier/normal_child_exit.json')['exit_code']==1
    f=r/'attempt/resources.jsonl';rr=[json.loads(line)for line in f.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
   elif version in('native_onpolicy_arm_collection_v2','native_onpolicy_arm_collection_v4','native_onpolicy_arm_collection_v5')and o.name=='val_000':
    import json
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    assert 'Sustained GPU utilization/VRAM threshold would not be met' in bind(r/'failure.json')['traceback']
    f=r/'attempt/resources.jsonl';rr=[json.loads(line)for line in f.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
   elif version=='native_onpolicy_arm_collection_v9'and o.name=='val_000':
    import json
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    assert 'Per-process cumulative/rolling GPU utilization' in bind(r/'failure.json')['traceback']
    f=r/'attempt/resources.jsonl';rr=[json.loads(line)for line in f.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
   elif version=='native_onpolicy_arm_collection_v10'and o.name=='val_000':
    import json
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    assert bind(r/'signal_received_21.json')['signal']==2 and bind(r/'attempt/barrier/normal_child_exit.json')['exit_code']==1
    receipt=r/'SIGINT_kernel_receipts.bin';assert receipt.stat().st_size==40;inputs[str(receipt)]=sha(receipt)
    f=r/'attempt/resources.jsonl';rr=[json.loads(line)for line in f.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
   elif version=='native_onpolicy_arm_collection_v17'and o.name=='val_000':
    import json
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    assert bind(r/'signal_received_21.json')['signal']==2 and bind(r/'attempt/barrier/normal_child_exit.json')['exit_code']==1
    receipt=r/'SIGINT_kernel_receipts.bin';assert receipt.stat().st_size==80;inputs[str(receipt)]=sha(receipt)
    f=r/'attempt/resources.jsonl';rr=[json.loads(line)for line in f.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
   elif version=='native_onpolicy_collection_v1'and o.name=='val_000':
    import json
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait'];failure=bind(r/'failure.json');assert 'Sustained GPU utilization/VRAM threshold would not be met'in failure['traceback']
    f=r/'attempt/resources.jsonl';rows=[json.loads(x)for x in f.read_text().splitlines()];assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
   elif version=='native_onpolicy_collection_v2':
    import json
    assert actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    failure=bind(r/'failure.json')['traceback'];assert 'Per-process cumulative/rolling GPU utilization' in failure or 'TimeoutExpired' in failure
    f=r/'attempt/resources.jsonl';rows=[json.loads(x)for x in f.read_text().splitlines()]
    assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f)
    # Completed expert screening/q0 only, never policy/masks/teacher labels.
    roots.append((r,spec))
   elif version=='native_onpolicy_collection_v3':
    import json
    assert o.name in('fit_001','val_000')and actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    if o.name=='val_000':
     f=r/'GPU_host.log';assert 'torch.OutOfMemoryError: CUDA out of memory'in f.read_text(errors='replace');inputs[str(f)]=sha(f)
    assert 'Per-process cumulative/rolling GPU utilization'in bind(r/'failure.json')['traceback']
    f=r/'attempt/resources.jsonl';rows=[json.loads(x)for x in f.read_text().splitlines()];assert rows[-1]['own_gpu_released']and not rows[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f)
    roots.append((r,spec))
   elif version=='native_onpolicy_collection_v8':
    import json
    assert o.name in('fit_002','val_000')and actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    failure=bind(r/'failure.json')['traceback'];assert 'Sustained GPU utilization/VRAM threshold would not be met'in failure or 'Per-process cumulative/rolling GPU utilization'in failure
    f=r/'attempt/resources.jsonl';records=[json.loads(x)for x in f.read_text().splitlines()];assert records[-1]['own_gpu_released']and not records[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f)
    roots.append((r,spec)) # Only complete expert validity/q0 inputs below.
   elif version=='native_onpolicy_collection_v9':
    import json
    assert o.name in('fit_002','val_000')and actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    failure=bind(r/'failure.json')['traceback']
    if o.name=='fit_002':assert 'TimeoutExpired'in failure and 'native_fast_inventory_v1/worker.py'in failure
    else:
     f=r/'GPU_host.log';assert 'RuntimeError: cannot create buffer'in f.read_text(errors='replace');inputs[str(f)]=sha(f)
    f=r/'attempt/resources.jsonl';records=[json.loads(x)for x in f.read_text().splitlines()];assert records[-1]['own_gpu_released']and not records[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
   elif version in('native_onpolicy_collection_v13','native_onpolicy_collection_v14','native_onpolicy_collection_v15','native_onpolicy_collection_v16'):
    import json
    assert o.name==('fit_003'if version=='native_onpolicy_collection_v13'else'fit_004')and actual['exit_code']==1 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
    if version=='native_onpolicy_collection_v16':
     log=r/'GPU_host.log';assert 'ValueError: Original TOPP gained mutable world dependencies'in log.read_text(errors='replace');inputs[str(log)]=sha(log)
    elif version=='native_onpolicy_collection_v15':
     log=r/'GPU_host.log';assert 'torch.OutOfMemoryError: CUDA out of memory'in log.read_text(errors='replace');inputs[str(log)]=sha(log)
    else:assert 'Sustained GPU utilization/VRAM threshold would not be met'in bind(r/'failure.json')['traceback']
    f=r/'attempt/resources.jsonl';rr=[json.loads(x)for x in f.read_text().splitlines()];assert rr[-1]['own_gpu_released']and not rr[-1]['remaining_own_gpu_pids'];inputs[str(f)]=sha(f);roots.append((r,spec))
 # Completed pure future expert/q0 envelopes remain valid preparation even
 # when their parent scientific full/label batch failed. Never import those labels.
 for source,spec in list(roots):
  future=source/'next_preparation';done=future/'preparation_complete.json'
  if done.exists()and future not in {r for r,_ in roots}:
   value=bind(done);assert value['complete']and value['required_future_fixed_dataset_work']and value['no_full_policy_outputs_masks_or_teacher_targets'];verify(value)
   for name,digest in value['input_sha256'].items():
    f=mapped(name);assert sha(f)==digest;inputs[str(f)]=digest
   roots.append((future,spec))
 cache={};requests={};keys={};diagnostics={};refs={};canonical=[]
 for task in tasks:
  state=dict(initial[task]);rows=[];items=[]
  while state['accepted']<limit:
   seed=state['next_seed'];found=[(r,spec,r/'screening'/task/(str(seed)+'.json'))for r,spec in roots if(r/'screening'/task/(str(seed)+'.json')).exists()]
   if not found:break
   r,spec,f=found[0];row=bind(f);assert row['task']==task and row['seed']==seed and row['episode_index']==state['accepted'];rows.append(row);canonical.append(str(f));state['next_seed']+=1
   if not row['eligible']:continue
   cid=task+':'+str(seed)
   if not(r/'accepted'/(cid+'.json')).exists()or not(r/'initial_prefetch'/(cid+'.json')).exists():
    rows.pop();canonical.pop();state['next_seed']-=1;break
   item=bind(r/'accepted'/(cid+'.json'));meta=bind(r/'initial_prefetch'/(cid+'.json'));req=bind(r/'semantic/requests'/(meta['content_key']+'.json'));frame=mapped(req['frame_file']);assert sha(frame)==req['frame_sha256'];inputs[str(frame)]=sha(frame)
   with np.load(frame,allow_pickle=False)as z:assert identity(key([z[f'cam{i}']for i in range(3)],z['command_state'],item['instruction']))==meta['content_key']==req['content_key']
   assert item['initial_physical_sha256']==row['initial_physical_sha256']and item['instruction']==req['instruction']and item['episode_index']==state['accepted']
   ref=r/'expert_references'/(cid+'.json')
   if not ref.exists():ref=mapped(spec['cached_preparation']['expert_reference_files'][cid])
   v=bind(ref);assert v['case_id']==cid and v['expert_plan_success']and v['expert_task_success']and v['sampling_stride_physics_steps']==15;raw=mapped(v['file']);assert sha(raw)==v['sha256'];inputs[str(raw)]=sha(raw)
   diagnostic=meta.get('diagnostic_identity')
   if diagnostic is None:
    actual=bind(r/'cases'/cid/'M_event_full/semantic_initial_receipt.json')['initial_diagnostic_identity'];diagnostic={k:actual[k]for k in('physical_sha256','rng_sha256')}
   assert set(diagnostic)=={'physical_sha256','rng_sha256'}
   items.append(item);state['accepted']+=1;keys[cid]=meta['content_key'];requests[req['content_key']]=req;diagnostics[cid]=diagnostic;refs[cid]=str(ref)
  cache[task]=dict(rows=rows,items=items,end_state=state)
 return dict(cached_valid_cases=sum(len(v['items'])for v in cache.values()),task_cache=cache,case_request_keys=keys,case_diagnostics=diagnostics,expert_reference_files=refs,requests=list(requests.values()),input_sha256=inputs,canonical_screen_files=canonical,no_policy_outputs_masks_or_labels_imported=True,historical_C5_preparation_recoveries_used=2,historical_C5_preparation_recoveries_maximum=2,new_automatic_recovery_attempts=0)
