"""Actual CPU forks verify exit/wait/proof order; no CUDA or model rollout."""
from pathlib import Path
import ast,copy,hashlib,json,os,select,tempfile,time
from .wait import wait_and_record
from .success import complete_and_exit
from native_nextchunk_v1.child_exit import identity,retired
T=Path(__file__).resolve().parents[1];O=T/'outputs/native_best100_exit_v1'
def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
def proof_order(code):
 with tempfile.TemporaryDirectory(prefix='cx_exit_cpu_')as d:
  d=Path(d);info_r,info_w=os.pipe();go_r,go_w=os.pipe();sync_r,sync_w=os.pipe();began=time.monotonic_ns();writer=os.fork()
  if writer==0:
   os.close(info_r);os.close(go_w);os.close(sync_w)
   child=os.fork()
   if child==0:
    os.close(info_w);os.close(sync_r);assert os.read(go_r,1)==b'1'
    (d/'whole_result').write_text('all fixture work completed')
    if code:os._exit(code)
    complete_and_exit()
   os.close(go_r);a,b=identity(child),identity(os.getpid());assert a and b
   os.write(info_w,(json.dumps(dict(child=a,parent=b))+'\n').encode())
   actual_fsync=os.fsync
   def blocked_fsync(fd):
    os.write(info_w,b'FSYNC_ENTERED\n');assert os.read(sync_r,1)==b'1';actual_fsync(fd)
   os.fsync=blocked_fsync
   got=wait_and_record(child,d/'exit.json',attempt_nonce='fixture_nonce',gpu_uuid='CPU_fixture_NOT_GPU',protocol_identity='fixture_identity',cpu_only_fixture=True)
   assert got==code
   os._exit(0)
  os.close(info_w);os.close(go_r);os.close(sync_r)
  with os.fdopen(info_r)as pipe:
   assert select.select([pipe],[],[],5)[0];data=json.loads(pipe.readline());a,b=data['child'],data['parent']
   verified={a['pid']:dict(host_pid=a['pid'],host_parent_pid=writer,child_started_ticks=a['started_ticks'],child_inner_pid=a['nspid'][-1],parent_inner_pid=b['nspid'][-1],parent_started_ticks=b['started_ticks'],pid_namespace=a['pid_namespace'],verified_at_ns=began)}
   os.write(go_w,b'1');os.close(go_w)
   assert select.select([pipe],[],[],5)[0]and pipe.readline().strip()=='FSYNC_ENTERED'
   # The writer is actually blocked inside fsync. The real waitpid proof must
   # already be visible, and the reaped child must be absent, while parent lives.
   path=d/'exit.json';record=json.loads(path.read_text());assert record['exit_code']==code and record['proc_absent_after_wait']and record['waited_child_inner_pid']==a['pid']
   assert not Path('/proc/'+str(a['pid'])).exists()and identity(writer)is not None
   expected=dict(attempt_nonce='fixture_nonce',gpu_uuid='CPU_fixture_NOT_GPU',protocol_identity='fixture_identity',cpu_only_fixture=True,attempt_started_ns=began)
   ids,note=retired(path,expected,verified)
   if code:assert not ids and not note['accepted']
   else:
    assert ids=={a['pid']}and note['accepted']
    for key,value in [('attempt_nonce','wrong'),('cpu_only_fixture',False),('gpu_uuid','wrong'),('protocol_identity','wrong')]:
     assert not retired(path,dict(expected,**{key:value}),verified)[0]
    assert not retired(path,expected,verified,pid_absent=lambda pid:False)[0]
    for field,value in [('child_started_ticks',a['started_ticks']+1),('exit_code',1),('normal_exit',False),('wait_status',256)]:
     bad=d/('bad_'+field+'.json');bad.write_text(json.dumps(dict(record,**{field:value})));assert not retired(bad,expected,verified)[0]
   os.write(sync_w,b'1');os.close(sync_w);waited,status=os.waitpid(writer,0);assert waited==writer and os.waitstatus_to_exitcode(status)==0
   assert not retired(path,expected,verified)[0], 'A departed or replaced CPU parent cannot authorize live retirement'
   assert(d/'whole_result').exists()
 return dict(actual_child_exit=code,writer_actual_exit=0,published_after_real_waitpid_before_fsync_completion=True,original_guard_verifier_used=True,fixture_only=True)
def teardown(direct):
 with tempfile.TemporaryDirectory(prefix='cx_exit_destructor_cpu_')as d:
  d=Path(d);pid=os.fork()
  if pid==0:
   class Owner:
    def __del__(self):
     (d/'destructor').write_text('python local destructor ran');time.sleep(.1)
   def run():
    owner=Owner();(d/'completed').write_text('fixture complete')
    if direct:complete_and_exit()
   run();os._exit(0)
  waited,status=os.waitpid(pid,0);assert waited==pid and os.waitstatus_to_exitcode(status)==0 and(d/'completed').exists()
  assert(d/'destructor').exists()==(not direct)
 return dict(direct=direct,actual_exit0=True,whole_fixture_work_committed=True,local_destructor_bypassed=direct)
def main():
 old=function(T/'native_nextchunk_v1/child_exit.py','wait_and_record');new=function(Path(__file__).with_name('wait.py'),'wait_and_record');assert ast.dump(old)==ast.dump(new)
 before=function(T/'native_best100_resume_v3/runtime.py','execute');after=function(T/'native_best100_resume_v4/runtime.py','execute')
 assert after.args.kwonlyargs[0].arg=='exit_success'and after.body[-2].value.func.id=='exit_success'
 after.args.kwonlyargs=[];after.args.kw_defaults=[];after.body=after.body[:-2];assert ast.dump(before)==ast.dump(after), 'Native scientific body/finally changed'
 result=dict(passed=True,actual_CPU_fork_wait_proofs=[proof_order(c)for c in(0,1)],teardown_checks=[teardown(x)for x in(False,True)],actual_waitpid_function_AST_unchanged=True,original_native_runtime_body_and_finally_AST_unchanged=True,wrong_attempt_reused_identity_live_child_failed_child_and_dead_parent_rejected=True,CPU_only=True,GPU_work=0,training_steps=0,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')})
 O.mkdir(exist_ok=True)
 with(O/'CPU_exit_handshake_checks.json').open('x')as f:json.dump(result,f,indent=2)
 print('ACTUAL_CPU_EXIT_HANDSHAKE_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
