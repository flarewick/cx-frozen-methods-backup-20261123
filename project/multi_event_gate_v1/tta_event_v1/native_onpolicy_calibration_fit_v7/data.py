"""Complete new325 and old100, before model/optimizer initialization."""
from pathlib import Path
import numpy as np,torch
from native_policy_execution_v1.common import T,read,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from native_onpolicy_calibration_data_v60.dataset import complete_data
from native_selected_outer_target_v1.data import inventory
from native_selected_outer_target_v1.qualified_reader import QualifiedSelectedReader
def load_all():
 file=T/'outputs/native_onpolicy_calibration_data_v60/qualified_dataset.json'
 if not file.exists():raise ValueError('Full325 qualified mixed dataset absent: model and optimizer forbidden')
 manifest=read(file);assert manifest['complete']and manifest['actual_build_host_exit0']and not manifest['actual_build_host_exit_pending'];verify(manifest)
 actual=mapped(manifest['build_host_exit_file']);assert sha(actual)==manifest['build_host_exit_sha256'];actual_exit=read(actual);assert actual_exit['exit_code']==0 and actual_exit['actual_Popen_wait_return']and actual_exit['proc_absent_after_wait']
 records,evidence=complete_data(include_arrays=True);assert evidence['groups']==manifest['groups']and evidence['identities']==325
 for row in records:
  for e in row['examples']:
   raw=e.pop('arrays');e['tensors']={k:torch.as_tensor(v,dtype=torch.int64 if k=='action_valid_offsets'else torch.float32)if isinstance(v,np.ndarray)else v for k,v in raw.items()}
 old_manifest,cases,plans,excluded=inventory();old=QualifiedSelectedReader();oldrows={}
 for plan in plans:
  p=plan['packet'];case=cases[p['case_id']];action=plan['alignment']['next_selected_at_or_after_feedback']['query'];value=old.packet_for_action(p,case['queries'],action);value['application_mode']='original_selected'
  row=oldrows.setdefault(p['case_id'],dict(case_id=p['case_id'],task=case['task'],split=p['split'],examples=[],old_original_case=True));row['examples'].append(dict(key=(p['case_id'],p['source_query'],p['query'],action),tensors=value))
 assert len(cases)==100 and sum(p['packet']['split']=='adapter_fit'for p in plans)==136 and sum(p['packet']['split']=='adapter_validation'for p in plans)==38 and len(excluded)==87
 records.extend(oldrows.values());bindings=dict(evidence['input_sha256']);bindings.update(old.observed_hashes);bindings[str(file)]=sha(file)
 return records,dict(new_population={k:v for k,v in evidence.items()if k!='input_sha256'},old_original_identities=100,old_FIT_packets=136,old_VAL_packets=38,old_retained_no_outer=len(excluded),input_sha256=bindings)
