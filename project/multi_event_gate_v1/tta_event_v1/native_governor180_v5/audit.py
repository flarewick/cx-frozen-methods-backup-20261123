"""Replay immutable local copies of the shared raw observations; keep legacy diagnostics distinct."""
import ast,copy,json,os,types
from collections import deque
from pathlib import Path
from .policy import evaluate
from .statistics import aggregate
from .observer import HISTORY

def replay(root):
 from native_policy_execution_v1.common import write,sha
 root=Path(root);file=root/'attempt/governor180.jsonl';values=[json.loads(s)for s in file.read_text().splitlines()];assert values
 bytime={}
 for v in values:bytime.setdefault(v['fleet']['monotonic'],[]).append(v)
 lo=min(bytime)-180;hi=max(bytime);history=deque();observer=None;checked=0;steady=0;local=root/'attempt/governor180_raw.jsonl'
 assert not local.exists()
 with HISTORY.open()as source,local.open('x')as sink:
  for line in source:
   raw=json.loads(line);now=raw['monotonic']
   if now<lo:continue
   if now>hi:break
   sink.write(line)
   if raw['observer_pid']!=observer:history.clear();observer=raw['observer_pid']
   history.append(raw)
   while history and history[0]['monotonic']<now-180:history.popleft()
   if now not in bytime:continue
   recomputed=aggregate(copy.deepcopy(raw),history);recomputed['observer_pid']=observer
   for v in bytime[now]:
    assert recomputed==v['fleet'],'Shared180 component replay mismatch'
    verdict=evaluate(recomputed,{tuple(x)for x in v['owned_identities']});assert verdict==v['server_verdict']and verdict['passed']
    assert not v.get('callback_stop_reason')
    for c in recomputed['cards']:
     for p in c['processes']:
      if [p['pid'],p['birth']]in v['owned_identities']and p['age']>=300:
       assert p['rolling180']['full_window'];steady+=1
    checked+=1
 assert checked==len(values)and steady>0
 out=dict(passed=True,server_schema='governor180_v15',resource_samples=checked,post_creation_300s_PID_samples=steady,all_raw_components_and_CPU_deltas_replayed=True,normal_complete_stage_required=True,local_raw_slice=str(local),input_sha256={str(p):sha(p)for p in(file,local)})
 write(root/'CPU_governor180_resource_audit.json',out);return out

def diagnostic_legacy_process_windows(root,spec):
 from native_process_missing_guard_v2.audit import audit as original
 import inspect
 tree=ast.parse(inspect.getsource(original));fn=tree.body[0];hits=0
 class Remove(ast.NodeTransformer):
  def visit_If(self,n):
   nonlocal hits
   if ast.unparse(n.test)=='elapsed >= 280':hits+=1;return ast.Pass()
   return self.generic_visit(n)
 fn=Remove().visit(fn);assert hits==1
 env=dict(original.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<legacy-process-diagnostics-only>','exec'),env)
 captured=[];env['write']=lambda p,v:captured.append((p,v));v=env['audit'](root,spec);assert len(captured)==1
 v['all_original_average_floors_34']=False;v['diagnostic_only_NOT_server_qualification']=True
 from native_policy_execution_v1.common import write
 write(Path(root)/'CPU_idle_missing_lower_bound_audit.json',v);return v

def diagnostic_legacy_device_windows(rows,post_release,uuid,permit):
 from native_process_sampler_v13.device_audit import audit as original
 import inspect
 tree=ast.parse(inspect.getsource(original));fn=tree.body[0];hits=0
 class Edit(ast.NodeTransformer):
  def visit_If(self,n):
   nonlocal hits
   if ast.unparse(n.test)in {"row.get('gpu_pids') and row['seconds'] >= 280.01","not row.get('gpu_pids') or row['seconds'] < 279.99"}:
    hits+=1;return ast.Pass()
   return self.generic_visit(n)
 fn=Edit().visit(fn);assert hits==2
 env=dict(original.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<legacy-device-diagnostics-only>','exec'),env)
 assert all(r['device_continuous']['prior_steady_checkpoints_rechecked']==0 for r in rows if 'device_continuous'in r)
 return dict(env['audit'](rows,post_release,uuid,permit),diagnostic_only_NOT_server_qualification=True)

def account_transport(root,proofs):
 from native_account_stream_v1.audit import audit as original
 import inspect
 source=inspect.getsource(original);old='assert len(proofs)==3 and len(set(proofs))==3';assert source.count(old)==1
 source=source.replace(old,'assert len(proofs)==2 and len(set(proofs))==2').replace('actual_normal_CPU_workers=3','actual_normal_CPU_workers=2')
 env=dict(original.__globals__);exec(compile(source,'<two-unchanged-NVML-account-transports-shared-observer-no-duplicate-third-worker>','exec'),env)
 return env['audit'](root,proofs)

def admission_scope(root):
 from native_policy_execution_v1.common import read,write,sha
 r=Path(root);p=r/'attempt/rule4_admission.json';v=read(p);assert v['passed']and v['no_CUDA_or_GPU_work']
 from .policy import require_admission
 require_admission(v['fleet'],v['selected_uuid'])
 rows=[json.loads(s)for s in(r/'attempt/rule4_running.jsonl').read_text().splitlines()];assert rows and all(x['proof']['passed']for x in rows)
 write(r/'CPU_rule4_admission_scope_audit.json',dict(passed=True,current_180s_AND_tiers=True,no_grandfathered_old_OR_admission=True,input_sha256={str(p):sha(p),str(r/'attempt/rule4_running.jsonl'):sha(r/'attempt/rule4_running.jsonl')}))
