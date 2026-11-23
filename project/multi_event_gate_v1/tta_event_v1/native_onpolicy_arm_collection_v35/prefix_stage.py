from .common import T,read,mapped,verify

def qualified(split,number):
 folder=T/'outputs/native_onpolicy_arm_collection_v35'/(split+'_'+str(number).zfill(3));actual=read(folder/'actual_host_wait_exit.json');assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 root=mapped(read(folder/'execution_state.json')['root']);verify(read(root/'protocol.json'));assert read(root/'atomic_full_recovery_qualification.json')['passed'];return folder
