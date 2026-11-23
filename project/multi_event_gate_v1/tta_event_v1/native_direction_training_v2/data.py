"""Same packet reader and explicit all100 identity/provenance qualification."""
from native_direction_training_v1.data import T,ARM,TASKS,CANDIDATE,KERNEL,read,sha,PacketReader
from pathlib import Path
from native_direction_cache_contract_v2.coverage import validate,POLICY

def full_manifest(path):
 path=Path(path).resolve();assert path.is_relative_to((T/'outputs').resolve())
 d=read(path)
 assert d['complete']and d['full_original_expert_cache_ready']and d['all_actual_issuing_query_hidden_bound']and d['actual_phase95_plus025_and_single_use_verified']
 assert d['candidate_identity']==CANDIDATE and d['original80FIT20VAL_split_unchanged']and d['all_original_cases_processed_without_outcome_filter']
 assert d['missing_coverage_policy']==POLICY and not d['training_authorized_by_this_artifact']
 for field in ('full_packet_manifest','coverage_and_packet_audit','actual_packet_build_and_audit_host_exit'):
  assert sha(d[field])==d[field+'_sha256']
 raw=read(d['full_packet_manifest']);audit=read(d['coverage_and_packet_audit']);host=read(d['actual_packet_build_and_audit_host_exit'])
 assert raw['complete']and raw['packets']==d['packets']and raw['cases']==d['coverage']
 assert audit['passed']and audit['all_packets_frozen_reader_verified']and audit['all100_identities_preserved']and audit['all_planned_streams_processed']and audit['original_recipe_minimum_packet_support']
 assert audit['manifest_sha256']==d['full_packet_manifest_sha256']
 assert host['exit_code']==0 and host['actual_build_and_audit_exit0']and host['completed_stage_codes']==[0,0]
 originalfile=T/'outputs/native_expert_full_inputs_v1/manifest.json';original=read(originalfile)
 assert d['original100_manifest']==str(originalfile)and d['original100_manifest_sha256']==sha(originalfile)
 indexfile=T/'outputs/native_expert_full_feedback_v1/index.json';index=read(indexfile)
 assert audit['index_sha256']==sha(indexfile)
 evidence=validate(original,index,d['coverage'],d['packets'],read,sha)
 assert evidence==d['missing_coverage_report']
 cases={r['case_id']:r for r in d['cases']};assert len(cases)==len(d['cases'])==100
 for c in index['cases']:assert cases[c['case_id']]==dict(c,feedback_processing_complete=True)
 pools={task:{split:[]for split in ('adapter_fit','adapter_validation')}for task in TASKS}
 for p in d['packets']:pools[cases[p['case_id']]['task']][p['split']].append(p)
 for task in TASKS:
  assert len(pools[task]['adapter_fit'])>=8 and pools[task]['adapter_validation']
  for group in pools[task].values():group.sort(key=lambda p:(p['original_expert_index'],p['source_query'],p['query']))
 for group in ('source_sha256','input_sha256'):
  for p,h in d[group].items():assert sha(p)==h,('full_cache_source_changed',p)
 return d,cases,pools
