"""Read-only whole qualified stage lookup; failed/partial attempts excluded."""
from .common import T,read,mapped,verify
def qualified(split,number):
 assert split in ('fit','val')and 0<=number<(8 if split=='fit'else 2)
 name=split+'_'+str(number).zfill(3)
 for version in ('native_onpolicy_collection_v24','native_onpolicy_collection_v21','native_onpolicy_collection_v20','native_onpolicy_collection_v19','native_onpolicy_collection_v18','native_onpolicy_collection_v17','native_onpolicy_collection_v16','native_onpolicy_collection_v15','native_onpolicy_collection_v14','native_onpolicy_collection_v13','native_onpolicy_collection_v12','native_onpolicy_collection_v11','native_onpolicy_collection_v10','native_onpolicy_collection_v9','native_onpolicy_collection_v8','native_onpolicy_collection_v7','native_onpolicy_collection_v1'):
  folder=T/'outputs'/version/name
  if not(folder/'actual_host_wait_exit.json').exists():continue
  actual=read(folder/'actual_host_wait_exit.json')
  if actual['exit_code']!=0:continue
  assert actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
  root=mapped(read(folder/'execution_state.json')['root']);proof=read(root/'atomic_full_recovery_qualification.json');assert proof['passed'];verify(proof);verify(read(root/'protocol.json'))
  return folder
 raise ValueError('Prior entire stage not qualified: '+name)
