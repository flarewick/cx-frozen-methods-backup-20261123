"""Freeze12 extra original-FIT developer q0 captures without outcome selection."""
from .paths import *
def prepare():
 designfile=T/'outputs/native_visible_expansion_v1/design.json';design=read(designfile);assert design['frozen']and len(design['new_scenes'])==12
 oldfile=T/'outputs/native_expert_full_cache_v8/protocol.json';old=read(oldfile)
 placement=O/'cpu_placement_plan.json';device=O/'device_plan.json';assert read(placement)['gpu_uuid']==read(device)['gpu_uuid']
 sources=dict(old['source_sha256'])
 for p in C.rglob('*'):
  if p.is_file()and p.suffix in('.py','.sh'):sources[str(p.relative_to(P))]=sha(p)
 for name in ('native_resource_policy_v4','native_fast_inventory_v1'):
  for p in (T/name).glob('*'):
   if p.is_file()and p.suffix in('.py','.json'):sources[str(p.relative_to(P))]=sha(p)
 for p in (T/'native_expert_smoke_v2/observe.py',T/'native_initial_frame_v3/observe.py'):sources[str(p.relative_to(P))]=sha(p)
 prior=T/'outputs/native_visible_expand_initial_v1';runs=list(prior.glob('host_run_*'));assert len(runs)==1 and read(runs[0]/'exit.json')['exit_code']==1
 assert not list(prior.glob('collections/*/case*/capture.json'))
 priorattempt=list(prior.glob('collections/*/host_attempts/*'));assert len(priorattempt)==1
 release=[__import__('json').loads(l)for l in(priorattempt[0]/'resources.jsonl').read_text().splitlines()];assert release[-1]['own_gpu_released']
 inputs={str(p):sha(p)for p in(oldfile,designfile,placement,device,T/'native_terminal_guard_v1/strict_guard.py',runs[0]/'exit.json',priorattempt[0]/'console.log',priorattempt[0]/'barrier/normal_child_exit.json',priorattempt[0]/'resources.jsonl')}
 spec=dict(version='native_visible_expand_initial_v2',scenes=design['new_scenes'],source_sha256=sources,input_sha256=inputs,runtime_source=old['runtime_source'],required_gpu_uuid=read(device)['gpu_uuid'],CPU_placement=str(placement),gpu_deadline_seconds=240,CPU_threads=2,initialization_eval_mode=True,one_time_warmup_per_task=True,purpose='Expanded fixed candidate developer evaluation-mode q0; no expert masks',native_initialization_failures_retained=True,no_scene_replacements=True,no_VLA_or_semantic_model_loaded=True,policy_actions=0,new_independent_scenes=0,online_updates=0,sustained_resource_qualification_claimed=False,TTA_utility_proven=False)
 spec['identity']=identity(spec);dest=O/'protocol.json'
 if dest.exists():assert read(dest)==spec
 else:write(dest,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args();s=prepare();root,m=dataset(s,a.gpu_uuid)
 print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 12)),flush=True)
