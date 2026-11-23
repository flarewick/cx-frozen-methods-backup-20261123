"""Actual Popen retention, unchanged CPU state, and permit/resource refusals."""
from pathlib import Path
import json,os,subprocess,sys,time,uuid
from native_policy_execution_v1.common import T,read,write,sha
from .wait import observe,audit

def run():
 root=T/'outputs/native_retained_cpu_wait_v1'/('CPU_fixture_'+uuid.uuid4().hex);root.mkdir(parents=True)
 source=Path(__file__).with_name('fixture.py');write(root/'protocol.json',dict(CPU_only_fixture=True,source_sha256={str(source):sha(source)}))
 sibling=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(45)'])
 log=(root/'CPU_child.log').open('x');process=subprocess.Popen([sys.executable,'-B',str(source),str(root)],stdout=log,stderr=subprocess.STDOUT)
 try:
  began=time.monotonic()
  while not(root/'source_bound_CPU_preload_proof.json').exists():
   assert process.poll()is None,(root/'CPU_child.log').read_text()
   assert time.monotonic()-began<40;time.sleep(.02)
  ticks=int(Path('/proc',str(process.pid),'stat').read_text().split(') ',1)[1].split()[19])
  write(root/'attempt/launch.json',dict(pid=process.pid,started_ticks=ticks,CPU_only_fixture=True))
  fleet=dict(CPU_only_fixture=True,own_gpu_count=0,cards=[dict(index=0,pids=[sibling.pid],own_pids=[])])
  for _ in range(3):
   value=observe(root,0,fleet,process);assert value['prepared_pid']==process.pid and process.poll()is None and sibling.poll()is None
  try:audit(root)
  except AssertionError:pass
  else:raise AssertionError('Audit accepted without permit boundary')
  write(root/'attempt/barrier/permit.json',dict(CPU_only_fixture=True,no_GPU_allocation=True))
  try:observe(root,0,fleet,process)
  except AssertionError:pass
  else:raise AssertionError('CPU wait accepted after permit')
  assert audit(root)['passed']
  boundary=root/'CPU_resource_refusal';write(boundary/'attempt/resources.jsonl',dict(CPU_only_fixture=True))
  try:observe(boundary,0,fleet,process)
  except AssertionError:pass
  else:raise AssertionError('CPU wait accepted after resource samples')
  write(root/'finish_CPU_wait.json',dict(finished=True))
  code=process.wait(timeout=10);assert code==0,(root/'CPU_child.log').read_text()
  assert read(root/'CPU_before.json')==read(root/'CPU_after.json')and sibling.poll()is None
 finally:
  if process.poll()is None:process.terminate()
  process.wait();log.close()
  if sibling.poll()is None:sibling.terminate()
  sibling.wait()
  write(root/'actual_CPU_exit.json',dict(exit_code=process.returncode,actual_Popen_wait_return=True,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists()))
  write(root/'actual_sibling_exit.json',dict(exit_code=sibling.returncode,actual_Popen_wait_return=True,proc_absent_after_wait=not Path('/proc',str(sibling.pid)).exists(),sibling_preserved_through_retention=True))
 files=[p for p in root.rglob('*')if p.is_file()]
 return dict(passed=True,actual_same_CPU_process_retained=True,CPU_tensor_numpy_and_RNG_byte_exact=True,
  unrelated_sibling_preserved=True,permit_or_GPU_resource_boundary_refused=True,actual_CPU_wait_exit0=True,
  no_GPU_or_native_qualification=True,input_sha256={str(p):sha(p)for p in files})
