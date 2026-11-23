"""Freeze new shadow/TTA/wrong development episodes with whole old full/zero controls."""
from .paths import *
CONDITIONS=('shadow_update','TTA','wrong_feedback')
def mapped(p):
 p=Path(p)
 if p.exists():return p
 for root in (Path('/data/qiufw/cx_migration/cx_new_direction_guide_TTA'),Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')):
  if p.is_relative_to(root):return P/p.relative_to(root)
 return p
def prepare():
 from native_trained_runtime_v1.paths import verify as oldverify
 oldfile=T/'outputs/native_trained_runtime_v1/protocol.json';old=read(oldfile);oldverify(old)
 terminalfile=T/'outputs/native_trained_admission_resume_v1/terminal_audit.json';terminal=read(terminalfile)
 assert terminal['passed']and terminal['actual_host_exit_code']==terminal['actual_child_exit_code']==0 and terminal['own_GPU_released']
 for p,h in terminal['input_sha256'].items():assert sha(mapped(p))==h
 oldroot=mapped(terminal['root']);oldcheck=read(oldroot/'CPU_episode_audit.json')
 assert oldcheck['passed']and oldcheck['all_requested_complete']and oldcheck['requested_branches']==20
 qfiles=list((T/'outputs/native_direction_visible_training_v1/runs').glob('*/qualification.json'));assert len(qfiles)==1
 qualification=qfiles[0];q=read(qualification)
 assert q['frozen']and q['actual_training_host_exit0']and q['actual_updates']==1000 and q['outer_target']=='next actual selected intervention, offline labels only'
 assert q['original_checkpoint_preserved']==old['checkpoint_sha256']
 device=read(O/'device_plan.json');placement=O/'cpu_placement_plan.json'
 assert device['gpu_uuid']==old['required_gpu_uuid']==read(placement)['gpu_uuid']
 # Existing full/zero action paths never consult learned response weights. Preserve
 # their original qualification and reuse whole complete episodes explicitly.
 controls=[];sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
 for p,h in terminal['input_sha256'].items():inputs[str(mapped(p))]=h
 for i,item in enumerate(old['scenes']):
  row=dict(case_id=item['case_id'],original_bundle=str(oldroot/('case%03d'%i)/'bundle.json'),conditions={})
  for condition in('M_event_full','zero_update'):
   folder=oldroot/('case%03d'%i)/condition
   assert read(folder/'result.json')['native_episode_complete']
   files={str(p):sha(p)for p in folder.rglob('*')if p.is_file()};assert files
   inputs.update(files);row['conditions'][condition]=dict(folder=str(folder),source_qualification=old['trained_qualification'],whole_complete_episode=True,file_sha256=files)
  controls.append(row)
 for name in('native_visible_runtime_v1','native_visible_episode_v1','native_visible_response_bridge_v1','native_action_visible_response_v1','native_resource_policy_v4','native_fast_inventory_v1'):
  for p in(T/name).rglob('*'):
   if p.is_file()and p.suffix in('.py','.sh','.json'):sources[str(p.relative_to(P))]=sha(p)
 failurefile=T/'outputs/native_selected_failure_v1/report.json';failure=read(failurefile)
 assert failure['actual_host_exit_code']==1 and failure['GPU_release']['own_gpu_released']and failure['whole_scene_bundles']==0 and not failure['source_resource_qualification_passed']
 inventorycheck=T/'outputs/native_fast_inventory_v1/CPU_actual_inventory_checks.json';assert read(inventorycheck)['passed']
 selectedterminal=T/'outputs/native_selected_runtime_v2/terminal_audit.json';assert read(selectedterminal)['passed']
 bridgecheck=T/'outputs/native_visible_response_bridge_v1/CPU_qualified_direction_bridge_checks.json';assert read(bridgecheck)['passed']
 loadercheck=qualification.parent/'CPU_qualified_loader_checks.json';assert read(loadercheck)['passed']
 contractcheck=O/'CPU_source_contract_checks.json';assert read(contractcheck)['passed']
 assert q['model_kind']=='ARM_rowspace_fast_increment_v1'and q['ARM_visible_interface_version']==1
 for file in(oldfile,terminalfile,qualification,Path(q['training_checkpoint']),Path(q['training_protocol']),Path(q['training_result_file']),placement,O/'device_plan.json',failurefile,inventorycheck,selectedterminal,bridgecheck,loadercheck,contractcheck):
  inputs[str(file)]=sha(file)
 spec=dict(old);spec.update(version='native_visible_runtime_v1',conditions=list(CONDITIONS),requested_native_episodes=12,comparison_total_branches=20,source_sha256=sources,input_sha256=inputs,CPU_placement=str(placement),trained_qualification=str(qualification),checkpoint_sha256=q['training_checkpoint_sha256'],previous_candidate_checkpoint_sha256=old['checkpoint_sha256'],reused_controls=controls,reused_control_protocol=str(oldfile),reused_control_protocol_identity=old['identity'],reused_control_terminal_audit=str(terminalfile),reused_control_scope='Whole M_event_full and zero_update on same server/GPU/scene/input. Old qualification preserved, weights unused by these actions. New shadow/TTA/wrong all freshly executed; no partial episode splice.',new_training_steps=1000,new_training_recipe='only fast hidden increment projected to ARM rowspace; same selectedouter/architecture/physicalkernel/seed17/fixed1000',resource_admission='v3: genuinely empty selected GPU and account<3; no other-own-card75/50 gate',model_kind=q['model_kind'],ARM_visible_projection_sha256=q['ARM_visible_projection_sha256'],fixed_projection_and_marker_required=True,TTA_utility_proven=False)
 spec.update(previous_monitoring_failure_report=str(failurefile),monitoring_change='v4 complete fresh NVML fleet, same3s fail-closed, sameotherguardAST',failed_partial_scene_or_branch_reuse=False,all12branches_reexecuted=True)
 spec.pop('identity');spec['identity']=identity(spec);dest=O/'protocol.json'
 if dest.exists():assert read(dest)==spec,'Frozen new candidate runtime changed'
 else:write(dest,spec)
 verify(spec);return spec
if __name__=='__main__':
 import argparse,json
 p=argparse.ArgumentParser();p.add_argument('--gpu-uuid',required=True);a=p.parse_args();s=prepare();root,m=dataset(s,a.gpu_uuid)
 print('FACTOR_PLAN',json.dumps(dict(root=str(root),identity=m['identity'],remaining=0 if(root/'result.json').exists()else 12)),flush=True)
