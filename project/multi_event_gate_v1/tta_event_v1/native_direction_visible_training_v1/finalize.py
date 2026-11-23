"""Freeze typed candidate only after actual fixed1000 host0 and loader checks."""
from pathlib import Path
import argparse,torch
from .common import *
def finalize(folder):
 folder=Path(folder).resolve();assert folder.is_relative_to(O.resolve())
 exfile=folder/'exit.json';ex=read(exfile);assert ex['actual_Popen_wait_return']and ex['exit_code']==0
 runs=list((O/'runs').iterdir());assert len(runs)==1;run=runs[0];r=read(run/'result.json');protocol=read(run/'protocol.json');coherencefile=run/'postfit_coherence.json';coherence=read(coherencefile)
 assert r['complete']and r['actual_updates']==1000 and r['CPU_only']and not r['CUDA_initialized']and r['zero_fast_prediction_and_action_exact']and r['postfit_action_visible_coherence_passed']
 assert coherence['passed']and coherence['original_selected_packets']==174 and coherence['optimizer_steps']==0
 assert sha(r['checkpoint'])==r['checkpoint_sha256']and sha(run/'protocol.json')==r['protocol_sha256']
 for group in('source_sha256','input_sha256'):
  for p,h in protocol[group].items():assert sha(p)==h
 auth=read(T/'outputs/native_feedback_confirm_v5/direction_training_authorization.json');report=read(auth['independent_feedback_report'])
 assert auth['direction_training_allowed']and report['complete']and report['independent_feedback_gate_passed']
 from .factory import model_for
 from native_visible_response_bridge_v1.bridge import tensor_sha,load_model
 initial,_,_=model_for(auth)
 q=dict(frozen=True,model_kind=initial.model_kind,ARM_visible_interface_version=1,ARM_visible_projection_sha256=tensor_sha(initial.ARM_visible_projection),awaiting_actual_host_exit0=False,actual_training_host_exit0=True,direction_trained=True,independent_feedback_gate_passed=True,
  independent_feedback_report=auth['independent_feedback_report'],independent_feedback_report_sha256=auth['independent_feedback_report_sha256'],
  training_checkpoint=r['checkpoint'],training_checkpoint_sha256=r['checkpoint_sha256'],training_protocol=str(run/'protocol.json'),training_protocol_sha256=sha(run/'protocol.json'),training_result_file=str(run/'result.json'),training_result_sha256=sha(run/'result.json'),training_host_exit_file=str(exfile),training_host_exit_sha256=sha(exfile),response_state_sha256=r['response_state_sha256'],source_sha256=protocol['source_sha256'],actual_updates=1000,
  trained_parameters='B_theta only; projector and marker immutable buffers',projection_only_fast_increment=True,original_static_hidden_conditioning_preserved=True,outer_target='next actual selected intervention, offline labels only',original100_identities_preserved=True,FIT_packets=136,VAL_packets=38,retained_no_future_packets=87,
  previous_selected_checkpoint_preserved='9acf416f3abe1a4af4445daa84e0f9bfd280487c4efa3d263ade7d2fa0993420',original_checkpoint_preserved='7392d9bbc8cdb293ed64f84d5fc9984824d531ef274f9de4f08b675f38f2dc8a',training_started_from_scratch=True,validation_report_only=True,checkpoint_selection=False,postfit_coherence_file=str(coherencefile),postfit_coherence_sha256=sha(coherencefile),independent_feedback_gate_scope='old physical kernel feedback evidence only; not projected update/action/task utility',new_independent_scenes=0,independent_action_and_task_benefit_still_required=True,TTA_utility_proven=False)
 draft=run/'qualification_draft.json';write(draft,q);model,_=load_model(draft)
 from native_trained_response_bridge_v1.bridge import load_model as old_loader
 try:old_loader(draft)
 except RuntimeError as exc:assert 'ARM_visible_interface_version'in str(exc)
 else:raise AssertionError('Old loader accepted projected response')
 oldfile=next((T/'outputs/native_direction_selected_training_v1/runs').glob('*/qualification.json'))
 try:load_model(oldfile)
 except ValueError as exc:assert 'Unprojected manifest'in str(exc)
 else:raise AssertionError('New loader accepted unprojected qualification')
 assert not torch.cuda.is_initialized()
 write(run/'CPU_qualified_loader_checks.json',dict(passed=True,new_typed_loader_accepted_actual_host0_candidate=True,old_loader_rejected_new_checkpoint_marker=True,new_loader_rejected_old_unprojected_qualification=True,model_kind=model.model_kind,actual_training_updates=1000,checkpoint_sha256=r['checkpoint_sha256'],new_independent_scenes=0,TTA_utility_proven=False))
 write(run/'qualification.json',q)
 print('ACTION_VISIBLE_ACTUAL_HOST0_CANDIDATE_FROZEN',run/'qualification.json',r['checkpoint_sha256'],flush=True)
 return q
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--host-run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_float32_matmul_precision('high');finalize(a.host_run)
