"""CPU-only reassign eligibility must never cancel a GPU-started attempt."""
from pathlib import Path
import json,tempfile,copy
from .admission_reassign import eligible
from .controller import O,save

def main():
 with tempfile.TemporaryDirectory(prefix='cx_cpu_admission_reassign_')as d:
  r=Path(d);(r/'attempt/barrier').mkdir(parents=True)
  (r/'protocol.json').write_text(json.dumps(dict(identity='bound',required_gpu_uuid='occupied')))
  (r/'source_bound_CPU_preload_proof.json').write_text(json.dumps(dict(passed=True,cpu_fork_child_passed=True,cuda_initialized=False,protocol_identity='bound')))
  fleet=dict(own_gpu_count=2,cards=[dict(uuid='occupied',available=False,pids=[77]),dict(uuid='other',available=True,pids=[])])
  read=lambda p:json.loads(p.read_text());other=lambda pid:'another.service'
  assert eligible(r,fleet,read,other)[0]
  for name in('attempt/barrier/permit.json','attempt/barrier/cuda_ready.json','attempt/resources.jsonl'):
   p=r/name;p.write_text('{}');assert not eligible(r,fleet,read,other)[0];p.unlink()
  assert not eligible(r,dict(fleet,own_gpu_count=3),read,other)[0]
  f=copy.deepcopy(fleet);f['cards'][1]['available']=False;assert not eligible(r,f,read,other)[0]
  f=copy.deepcopy(fleet);f['cards'][0]['available']=True;f['cards'][0]['pids']=[];assert not eligible(r,f,read,other)[0]
  assert not eligible(r,fleet,read,lambda pid:'cx-tta-frozen100-v6-'+__import__('os').environ.get('CX_TTA100_COHORT','pads')+'.service')[0]
  p=r/'source_bound_CPU_preload_proof.json';v=read(p)
  for change in(dict(cuda_initialized=True),dict(protocol_identity='wrong'),dict(cpu_fork_child_passed=False)):
   p.write_text(json.dumps(dict(v,**change)));assert not eligible(r,fleet,read,other)[0]
 save(O/'CPU_admission_reassign_checks.json',dict(passed=True,CPU_only_other_job_took_card_can_reassign=True,GPU_permit_or_runtime_never_cancelled=True,account3_no_empty_card_own_occupant_and_stale_CPU_proof_rejected=True,GPU_started=False))
 print('SAFE_CPU_ONLY_ADMISSION_REASSIGN_CHECKS_PASS')
if __name__=='__main__':main()
