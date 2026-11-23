"""Keep the original telemetry/exit engine; replace only retired server decisions."""
import ast,copy,json,os,time
from pathlib import Path
from native_governor180_v4.policy import evaluate
from native_governor180_v4.observer import latest
from native_process_missing_guard_v2 import guard as original
_implementation=None

def base_transformed(source):
 from native_process_missing_guard_v2.rewrite import transform
 tree=transform(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='guarded_command')
 count={'floor':0,'old_resource':0,'old_steady':0,'device_diagnostic':0}
 class Replace(ast.NodeTransformer):
  def visit_If(self,n):
   test=ast.unparse(n.test)
   if test=='not 33 <= min_cumulative_utilization <= 100':
    count['floor']+=1;return ast.parse("if min_cumulative_utilization != 0: raise ValueError('Retired cumulative floor must be disabled; new180 policy enforces all live server thresholds')").body[0]
   if test=='elapsed >= 280 and active':count['old_resource']+=1;return ast.Pass()
   if test=='elapsed >= 300 and active and (steady_min_utilization is not None)':count['old_steady']+=1;return ast.Pass()
   return self.generic_visit(n)
  def visit_Call(self,n):
   if isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='device_window'and n.func.attr=='update':
    for k in n.keywords:
     if k.arg=='steady_required':k.value=ast.Constant(False);count['device_diagnostic']+=1
   return self.generic_visit(n)
 changed=Replace().visit(copy.deepcopy(fn));assert count==dict(floor=1,old_resource=1,old_steady=1,device_diagnostic=1),count
 return ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))

def drained(active,verified,retired,note):
 # `retired` is the original fresh nonce/namespace/birth/normal-wait validator,
 # not an inference from empty NVML samples or process creation time.
 return bool(not active and len(verified)==1 and set(verified)==set(retired)
             and note.get('accepted')is True and note.get('exit_code')==0
             and note.get('normal_child_reaped')is True
             and note.get('proc_absent_rechecked')is True
             and note.get('parent_alive_same_identity')is True
             and note.get('retired_host_pid')in retired)

def transformed(source):
 tree=base_transformed(source);counts=dict(retirement=0,priority=0,memory=0,telemetry=0)
 class Edit(ast.NodeTransformer):
  def visit_Assign(self,n):
   text=ast.unparse(n)
   if text=='active = [pid for pid in active if pid not in retired_children]':
    counts['retirement']+=1
    return[n,*ast.parse('post_GPU_drained = _drained(active,verified_gpu_pids,retired_children,child_exit_note)').body]
   if text.startswith('telemetry = '):
    assert isinstance(n.value,ast.Dict);n.value.keys.append(ast.Constant('GPU_drained_CPU_tail'));n.value.values.append(ast.Name(id='post_GPU_drained',ctx=ast.Load()));counts['telemetry']+=1
   return self.generic_visit(n)
  def visit_If(self,n):
   test=ast.unparse(n.test)
   if test=="priority['stop_for_prior_or_unverified_occupant']":
    n.test=ast.BoolOp(op=ast.And(),values=[n.test,ast.UnaryOp(op=ast.Not(),operand=ast.Name(id='post_GPU_drained',ctx=ast.Load()))]);counts['priority']+=1
   if test=='memory > max_memory_mib':
    n.test=ast.BoolOp(op=ast.And(),values=[n.test,ast.UnaryOp(op=ast.Not(),operand=ast.Name(id='post_GPU_drained',ctx=ast.Load()))]);counts['memory']+=1
   return self.generic_visit(n)
 changed=Edit().visit(tree);assert counts==dict(retirement=1,priority=1,memory=1,telemetry=1),counts
 return ast.fix_missing_locations(changed)

def implementation():
 global _implementation
 if _implementation is None:
  base=original._implementation();scope=dict(base.__dict__,_drained=drained)
  source=Path(__file__).resolve().parents[1]/'native_process_sampler_v13/strict_guard.py'
  exec(compile(transformed(source.read_text()),str(source)+'<server180-only-decisions>','exec'),scope)
  _implementation=(scope['guarded_command'],base)
 return _implementation

def guarded_command(*args,**kwargs):
 assert 'min_cumulative_utilization'not in kwargs and kwargs.get('steady_min_utilization')is None
 fn,base=implementation();process=kwargs['prepared_process'];report=Path(args[2]);callback=kwargs.get('on_sample');owned=set();pending={}
 def sample(row):
  table=base.process_table();tree=base.descendants(process.pid,table)
  owned.update((p,table[p]['started_ticks'])for p in tree if p in table)
  fleet=latest();verdict=evaluate(fleet,owned)
  observed={(p['pid'],p['birth'])for c in fleet['cards']for p in c['processes']}
  missing=[p for p in row['gpu_pids']if p in table and (p,table[p]['started_ticks'])not in observed]
  now=time.monotonic()
  for p in missing:pending.setdefault(p,now)
  for p in set(pending)-set(missing):pending.pop(p)
  if any(now-v>8 for v in pending.values()):verdict=dict(verdict,passed=False,reasons=verdict['reasons']+['Own GPU PID absent from shared observer for more than8s'])
  row['governor180']=dict(schema='governor180_v15',fleet=fleet,server_verdict=verdict,owned_identities=sorted(owned),pending_first_observation_pids=missing,retired_cumulative_and_300_server_decisions=True)
  reason='; '.join(verdict['reasons'])if verdict['reasons']else(callback(row)if callback else None)
  record=dict(row['governor180'],seconds=row['seconds'],gpu_pids=row['gpu_pids'],callback_stop_reason=reason)
  with report.with_name('governor180.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
  return reason
 base.gpu_doctor.report=report.with_name('NVML_queries.jsonl')
 from native_process_sampler_v13.metrics import sampling_session
 from native_account_stream_v1.transport import context
 with context(report.with_name('NVML_account_monitor_actual_exit.json')),base.gpu_doctor.context('monitor'),sampling_session(kwargs['gpu'],report.with_name('NVML_stream_actual_exit.json')):
  return fn(*args,**dict(kwargs,on_sample=sample,min_cumulative_utilization=0))

def __getattr__(name):return getattr(original,name)
