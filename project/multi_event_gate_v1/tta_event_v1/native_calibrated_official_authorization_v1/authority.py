"""Explicit user authorization for measuring this frozen candidate at official100."""
from native_policy_execution_v1.common import T,read,sha
AUTH=T/'outputs/native_calibrated_official_authorization_v1/user_official100_authorization.json'
TASKS={'open_laptop','grab_roller','beat_block_hammer','turn_switch','adjust_bottle'}

def validate(value,report,checkpoint):
 if not value:raise ValueError('Explicit official100 user instruction required')
 assert value['explicit_user_authorization']and value['override_positive_confirmation_gate_for_this_evaluation_only']
 assert value['scope']=='official_seed0_five_task_paired100_current_frozen_candidate'
 assert set(value['tasks'])==TASKS and value['episodes_each_task']==100 and value['conditions']==['baseline','M_event_full','TTA']
 assert value['official_seed']==0 and value['actual_scene_seed_start']==100000
 assert value['candidate_checkpoint_sha256']==checkpoint==report['candidate_checkpoint_sha256']
 assert report['complete']and report['all_actual_host_wait0']and report['interface_qualified']
 assert value['historical_confirmation_net_success_change']==report['paired_net_success_change']==-1
 assert value['historical_result_unchanged']and value['no_parameter_tuning_or_EVAL_fitting']
 return True

def require(report,checkpoint):
 value=read(AUTH);assert validate(value,report,checkpoint)
 return {str(AUTH):sha(AUTH),str(__file__):sha(__file__)}
