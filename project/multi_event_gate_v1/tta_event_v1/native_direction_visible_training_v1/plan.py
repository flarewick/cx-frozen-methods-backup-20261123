"""Previous candidate terminal and structural counterexample checked before new fit."""
from .common import *
def mapped(p):
 p=Path(p)
 if p.exists():return p
 for root in(Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA'),Path('/data/qiufw/cx_migration/cx_new_direction_guide_TTA')):
  if p.is_relative_to(root):return T.parents[1]/p.relative_to(root)
 return p
def prerequisites():
 from native_direction_selected_training_v1.plan import prerequisites as prior
 inputs=dict(prior()['input_sha256'])
 terminalfile=T/'outputs/native_selected_runtime_v2/terminal_audit.json';terminal=read(terminalfile)
 assert terminal['passed']and terminal['actual_host_exit_code']==terminal['actual_child_exit_code']==0 and terminal['own_GPU_released']and terminal['requested_branches']==12
 for p,h in terminal['input_sha256'].items():assert sha(mapped(p))==h
 reportfile=T/'outputs/native_selected_candidate_progress_v2/complete_negative_result.json';report=read(reportfile)
 assert report['complete']and report['candidate_rejected_for_efficacy']and report['TTA_rescues']==0 and report['TTA_harms']==2
 checkfile=T/'outputs/native_action_visible_response_v1/CPU_coherence_checks_v2.json';checks=read(checkfile)
 assert checks['passed']and checks['ARM_null_basis_probes']==832 and checks['old_unprojected_qualification_rejected']and checks['distinct_immutable_response_buffer_namespace']and checks['optimizer_steps']==0 and not checks['prototype_qualified_for_actor']
 for p,h in checks['source_sha256'].items():assert sha(mapped(p))==h
 nullfile=T/'outputs/native_action_null_coupling_audit_v1/report.json';null=read(nullfile)
 assert null['passed']and null['eligible_packets']==174 and null['globally_ARM_invisible_offset_dimension']==52
 for p in(terminalfile,reportfile,checkfile,nullfile):inputs[str(p)]=sha(p)
 return dict(input_sha256=inputs,new_candidate_reason='52 ARM-invisible offset dimensions can change proxy; only project fast increment to immutable W rowspace. Previous selected candidate1of4 vsfull3of4 rejected.',recipe=RECIPE,new_independent_scenes=0)
