"""Require actual host/child success and GPU release; never infer it from frames."""
from pathlib import Path
import json,hashlib
T=Path(__file__).resolve().parents[1];P=T.parents[1];O=T/'outputs/native_visible_expand_initial_v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hostpath(p):
 p=Path(p)
 if p.exists():return p
 canonical=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')
 return P/p.relative_to(canonical)
def audit():
 runs=list(O.glob('host_run_*'));assert len(runs)==1
 exitfile=runs[0]/'exit.json';ex=read(exitfile);assert ex['actual_Popen_wait_return']and ex['exit_code']==0
 specfile=O/'protocol.json';spec=read(specfile);roots=list((O/'collections').iterdir());assert len(roots)==1;root=roots[0]
 result=read(root/'result.json');complete=read(root/'GPU_frames_complete.json');frames=read(root/'CPU_frame_audit.json');manifest=read(root/'manifest.json')
 assert result['complete']and frames['passed']and complete['complete']
 assert result['protocol_identity']==complete['protocol_identity']==manifest['protocol_identity']==spec['identity']
 assert result['short_initialization_only']and not result['long_resource_qualification_claimed']
 attempts=list((root/'host_attempts').iterdir());assert len(attempts)==1;attempt=attempts[0]
 childfile=attempt/'barrier/normal_child_exit.json';child=read(childfile)
 assert child['normal_exit']and child['exit_code']==0 and child['proc_absent_after_wait']and not child['cpu_only_fixture']
 assert child['protocol_identity']==spec['identity']and child['gpu_uuid']==manifest['gpu_uuid']
 resourcefile=attempt/'resources.jsonl';resources=[json.loads(s)for s in resourcefile.read_text().splitlines()]
 assert not any('stopped'in r for r in resources)
 releases=[r for r in resources if'own_gpu_released'in r];assert len(releases)==1
 release=releases[0];assert release['own_gpu_released']and not release['remaining_own_gpu_pids']
 if Path('/proc/%d'%ex['host_pid']).exists():
  ticks=int(Path('/proc/%d/stat'%ex['host_pid']).read_text().split(') ',1)[1].split()[19]);assert ticks!=ex['host_started_ticks'],'Original host still live'
 for pid in {p for r in resources for p in r.get('gpu_pids',[])}:assert not Path('/proc/%d'%pid).exists(),'Former actual CUDA PID still present'
 files=[specfile,exitfile,childfile,resourcefile,root/'result.json',root/'GPU_frames_complete.json',root/'CPU_frame_audit.json',root/'manifest.json']
 for row in complete['rows']:
  if not row['frame_available']:
   assert row['initialization_failure']['type']=='UnStableError';continue
  p=hostpath(row['file']);assert sha(p)==row['sha256'];files.append(p)
 return dict(passed=True,actual_host_exit_code=0,actual_child_exit_code=0,own_GPU_released=True,new_independent_scenes=0,short_initialization_only=True,sustained_resource_qualification_claimed=False,root=str(root),input_sha256={str(p):sha(p)for p in files})
if __name__=='__main__':
 value=audit();dest=O/'terminal_audit.json'
 if dest.exists():assert read(dest)==value
 else:
  with dest.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print('ACTUAL_DEVELOPER_Q0_HOST_CHILD_RELEASE_PASS',flush=True)
