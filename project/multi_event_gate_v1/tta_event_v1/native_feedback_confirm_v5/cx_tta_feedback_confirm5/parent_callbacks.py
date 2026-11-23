"""CPU-only case validation/ordered commits, including two old coupled bridges."""
from pathlib import Path
from .paths import read,write,digest
from .contract import verify_case,commit,load_packet,same,BRANCHES
from .actor_trace import verify_trace
from .current_case import reference_equal
from .live_audit import verify_live
def callbacks(spec,root,manifest,journal,session,*,cpu_only_fixture=False):
 if cpu_only_fixture:
  assert manifest.get("cpu_only_fixture")is True and all(v["case_id"].startswith("CPU_fixture_")for v in spec["scenes"])
 initial=len(journal['cases'])
 def validate(local,folder):
  index=initial+local;item=spec['scenes'][index]
  assert read(folder/'case.json')['case_id']==item['case_id']
  audit=verify_case(folder,spec['projection'])
  trace=verify_trace(folder,item['instruction'])
  write(folder/'actor_model_trace_audit.json',trace)
  if not cpu_only_fixture:
   witness=read(folder/'cache_forward_witness.json')
   assert not witness['cpu_only_fixture']and witness['actual_owner_forward_calls']
   assert witness['case_id']==item['case_id']and sum(witness['base_epochs'].values())==trace['actual_model_queries']
  live_audits={}
  if audit['available']:
   for branch in BRANCHES:live_audits[branch]=verify_live(folder,branch,item,teacher=bool(item['teacher_records']),cpu_only_fixture=cpu_only_fixture)
   if item.get('requires_sequential_reference'):
    for branch in('Z_reference','C_original'):
     live_audits['sequential_'+branch]=verify_live(folder/'sequential_reference',branch,item,teacher=bool(item['teacher_records']),cpu_only_fixture=cpu_only_fixture)
   write(folder/'live_feedback_audit.json',dict(passed=True,branches=live_audits,CPU_parent_only=True))
  bridge=index<spec['bridge_count']
  if bridge:
   assert not cpu_only_fixture
   first=read(session/'first_CUDA_cache_reclamation.json')
   assert not first['cpu_only_fixture']and first['actual_unused_reserved_bytes_released']>0
   assert first['inner_pid']==read(session/'cpu_validation/parent_validator_ready.json')['child_pid']
   assert first['model_parameter_pointers_unchanged']and first['Torch_CPU_CUDA_RNG_unchanged']and first['participant_model_count']==2
   assert witness['post_reclamation_encoder_forward']and witness['post_reclamation_base_forward']
   if item['earliest_seed']['available']:assert witness['post_reclamation_tracker_forward']
  if bridge and audit['available']:
   reference=Path(item['bridge_reference']);checks=0
   for name in('proposal.npz','initial_current.npz','observed_history.npz','selected_query.npz'):
    assert same(load_packet(folder/name),load_packet(reference/name)),name;checks+=1
   for name in('actor_history.json','prefix_clock.json','selected_query.json','discovery.json'):
    assert read(folder/name)==read(reference/name),name;checks+=1
   for branch in BRANCHES:checks+=reference_equal(folder,reference,branch,legacy_prefix=not item.get('bridge_full_three_chunk_reference',False))
   proof=dict(passed=True,checks=checks,reference=str(reference),
       reference_case_evidence_sha256=item['bridge_evidence_sha256'],
       all_current_main_branch_numeric_RGB_physics_RNG_queries_receipts_exact=True,
       source_references_not_reexecuted=True,excluded_from_training=True)
   write(folder/'compatibility_bridge.json',proof)
  return dict(passed=True,audit=audit,compatibility_bridge_passed=bridge)
 def accepted(local,folder,result):
  index=initial+local;assert len(journal['cases'])==index
  row=commit(root,manifest,spec,journal,folder,session.name)
  assert row['audit']==result['audit']
  write(root/'validation_progress.json',dict(validated=len(journal['cases']),requested=len(spec['scenes']),
     last_case=row['case_id'],CPU_parent_only=True,complete=False))
  return row
 return validate,accepted
