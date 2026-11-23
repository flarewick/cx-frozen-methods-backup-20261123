"""CPU original full dispatch/resource contracts, plus actual packet index parity."""
from pathlib import Path
import json,torch
from .common import T,O,read,write,sha
from .contracts import check
from .index import index

def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();check()
 from native_feedback_application_v1.episode import execute
 from .episode import execute as bound
 assert execute is bound
 count=0;packets=0;targets=0
 for root in(T/'outputs/native_object_transfer_pilot_v3/development').glob('batches/*/runs/*'):
  if not(root/'terminal_audit.json').exists():continue
  spec=read(root/'protocol.json')
  for folder in root.glob('cases/*/M_event_full'):
   value=index(folder,spec['event_config']);count+=1;packets+=len(value['packets']);targets+=len(value['potential_actions'])
 assert count==20 and packets==16 and targets==15
 value=dict(passed=True,CPU_only=True,old20_development_read_only=True,old_packets_reconstructed=packets,old_potential_actions=targets,new_full_data_NOT_collected=True,no_teacher_labels_or_TRAIN_relabel=True,original_action_and_guard_contracts_passed=True,CUDA_initialized=False,optimizer_steps=0,source_sha256={str(f):sha(f)for f in Path(__file__).parent.rglob('*')if f.is_file()and f.suffix in('.py','.sh','.md')})
 write(O/'CPU_contract_checks_v3.json',value);print('ONPOLICY_FULL_CPU_CONTRACTS_PASS',flush=True)
if __name__=='__main__':main()
