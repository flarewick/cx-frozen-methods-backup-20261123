"""Read q0 at the original full entry; regenerate semantics only for new real RGB."""
import numpy as np
from .common import write,mapped,sha
from .semantic import content_key

def decision(initial,diagnostic,command,old_key,instruction):
 assert initial['native_action_count']==0
 assert {k:initial[k]for k in('physical_sha256','rng_sha256')}==diagnostic,'Cached q0 physical/RNG boundary changed'
 assert np.array_equal(initial['command_state'],command),'Cached q0 own command changed'
 actual=content_key(initial['rgb'],initial['command_state'],instruction)
 return actual,actual!=old_key

def prepare(task,item,cfg,raw,routed_seed,*,spec,meta,prepared,root,unstable,replay):
 attempts=1
 try:initial=replay(task,item,[],cfg,raw,routed_seed)
 except unstable:
  task.close_env(clear_cache=False);attempts+=1;initial=replay(task,item,[],cfg,raw,routed_seed)
 old=meta['content_key'];request=next(r for r in spec['cached_preparation']['requests']if r['content_key']==old);frame=mapped(request['frame_file']);assert sha(frame)==request['frame_sha256']
 with np.load(frame,allow_pickle=False)as z:actual,changed=decision(initial,meta['diagnostic_identity'],z['command_state'],old,item['instruction'])
 if changed:assert prepared['seed_store'].submit(initial,item['instruction'])==actual
 current=dict(meta,content_key=actual,actual_current_q0_revalidated=True,old_q0_key=old,old_RGB_exact=not changed,requires_whole_fresh_full_control=changed)
 write(root/'initial_prefetch'/(item['case_id']+'.json'),current)
 write(root/'cached_q0_revalidation'/(item['case_id']+'.json'),dict(case_id=item['case_id'],actual_native_setup_attempts=attempts,native_action_count=0,old_physics_RNG_command_exact=True,old_RGB_exact=not changed,current_content_key=actual,old_content_key=old,semantic_input_from_actual_current_scene=True,requires_whole_fresh_full_control=changed,no_extra_policy_queries_or_physics=True,no_old_masks_or_policy_outputs_or_labels_imported=True,actual_policy_entry_byte_check_pending=True))
 return initial,changed
