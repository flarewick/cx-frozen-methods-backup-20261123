from .common import ROOT,read,verify,sha

def audit(root,spec):
 assert root==ROOT
 p=root/'CPU_closed_expert_task_release_audit.json';v=read(p);verify(v)
 assert v['passed']and v['whole_native_reference_controls']==4 and v['all_cached_policy_entries_byte_exact']and v['all_closed_scene_and_task_and_proxy_weakrefs_dead']
 return v
