"""Four-row contract input: three old actual heads plus explicit missing fixture."""
from pathlib import Path
import numpy as np
from native_cpu_seed_v4.common import T,read,sha,write,rgb_sha,language_sha
def run():
 root=T/'outputs/native_cpu_seed_v4/developer_bridge';assert not root.exists()
 oldfile=T/'outputs/native_cpu_seed_v2/protocol.json';old=read(oldfile);jobs=[]
 for case in('stack_blocks_two:7103000','stack_bowls_two:7103000','stack_blocks_two:7103004'):
  job=next(r for r in old['jobs']if r['case_id']==case)
  assert sha(job['history'])==job['history_sha256']
  with np.load(job['history'],allow_pickle=False)as z:images=[z['cam'+str(i)][0].copy()for i in range(3)];state=z['states'][0].copy()
  root.mkdir(parents=True,exist_ok=True);file=root/('input%03d.npz'%len(jobs))
  with file.open('xb')as f:np.savez_compressed(f,**{'cam'+str(i):im for i,im in enumerate(images)},command_state=state)
  jobs.append(dict(index=len(jobs),case_id=case,available=True,query=0,tick=0,instruction=job['instruction'],
   file=str(file),sha256=sha(file),head_RGB_sha256=rgb_sha(images[0]),instruction_sha256=language_sha(job['instruction']),
   original_observed_history=job['history'],original_observed_history_sha256=job['history_sha256']))
 jobs.append(dict(index=3,case_id='CPU_missing_initial_fixture',available=False,reason='explicit_missing_q0_fixture_not_a_real_scene'))
 manifest=dict(version=1,kind='actual_q0_semantic_preparation',scope='CPU_developer_contract_bridge',requested=4,
  jobs=jobs,actual_old_inputs=3,missing_fixture=1,old_developer_protocol=str(oldfile),old_developer_protocol_sha256=sha(oldfile),
  new_independent_scenes=0,online_updates=0,TTA_utility_proven=False)
 write(root/'manifest.json',manifest);print('GENERIC_CPU_BRIDGE_MANIFEST_READY',len(jobs),flush=True)
if __name__=='__main__':run()
