"""Freeze a four-case five-condition cohort from fixed expanded developer design."""
from .paths import *
from native_visible_episode_v1.episode import CONDITIONS
def mapped(path):
 path=Path(path)
 if path.exists():return path
 for root in (Path('/data/qiufw/cx_migration/cx_new_direction_guide_TTA'),Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')):
  if path.is_relative_to(root):return P/path.relative_to(root)
 return path
COHORT='c'
def prepare():
 designfile=T/'outputs/native_visible_expansion_v1/design.json';design=read(designfile);cohort=next(c for c in design['cohorts']if c['name']==COHORT)
 assert design['frozen']and len(cohort['scenes'])==4 and design['requested_scenes']==16
 oldfile=T/'outputs/native_visible_runtime_v1/protocol.json';old=read(oldfile)
 qfile=mapped(design['candidate_qualification']);q=read(qfile);assert sha(qfile)==design['candidate_qualification_sha256']and q['training_checkpoint_sha256']==design['checkpoint_sha256']
 seedroot=T/'outputs/native_visible_expand_semantics_v2';seeds=read(seedroot/'result.json');assert seeds['complete']and seeds['requested']==12 and seeds['CPU_models_unchanged']and not seeds['CUDA_initialized']
 device=read(O/'device_plan.json');placement=O/'cpu_placement_plan.json';assert device['gpu_uuid']==read(placement)['gpu_uuid']
 sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
 for directory in (C,T/'native_visible_expansion_v1',T/'native_visible_expand_semantics_v2',T/'native_visible_expand_initial_v2'):
  for p in directory.rglob('*'):
   if p.is_file()and p.suffix in('.py','.sh','.json'):sources[str(p.relative_to(P))]=sha(p)
 for field in ('source_sha256','input_sha256'):
  for path,h in read(seedroot/'protocol.json')[field].items():assert sha(path)==h;inputs[path]=h
 for p in seedroot.iterdir():
  if p.is_file():inputs[str(p)]=sha(p)
 for row in seeds['rows']:
  for path in (row.get('frame_file'),row.get('mask_file')):
   if path is not None:inputs[path]=sha(path)
 terminal=T/'outputs/native_visible_runtime_v1/terminal_audit.json';assert read(terminal)['passed']
 for file in (designfile,oldfile,qfile,Path(q['training_checkpoint']),placement,O/'device_plan.json',terminal,O/'CPU_source_contract_checks_v6.json'):
  inputs[str(file)]=sha(file)
 for cert in O.glob('gpu-doctor-*.json'):inputs[str(cert)]=sha(cert)
 spec=dict(old)
 for key in ('identity','reused_controls','reused_control_protocol','reused_control_protocol_identity','reused_control_terminal_audit','reused_control_scope'):spec.pop(key,None)
 spec.update(version='native_visible_expand_c_v1',scope='expanded_original_FIT_developer_cases_only',cohort=COHORT,scenes=cohort['scenes'],fixed_case_ids=[s['case_id']for s in cohort['scenes']],conditions=list(CONDITIONS),requested_developer_scenes=4,requested_native_episodes=20,comparison_total_branches=20,simulation_lanes=4,required_gpu_uuid=device['gpu_uuid'],CPU_placement=str(placement),source_sha256=sources,input_sha256=inputs,trained_qualification=str(qfile),checkpoint_sha256=q['training_checkpoint_sha256'],new_training_steps=0,expanded_design_identity=design['identity'],new_independent_scenes=0,all20branches_freshly_executed=True,reused_whole_controls=0,resource_admission='v4 fresh complete NVML; genuinely empty any of7, account<3, all unchanged steadyguards',TTA_utility_proven=False)
 spec['identity']=identity(spec);dest=O/'protocol.json'
 if dest.exists():assert read(dest)==spec
 else:write(dest,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args();s=prepare();root,m=dataset(s,a.gpu_uuid)
 print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 20)),flush=True)
