"""Source parity and actual observed-window equivalence, CPU only."""
import ast,json,os,copy,hashlib
from pathlib import Path
from .observer import C,T,verify_reference
from .statistics import aggregate

def main():
 verify_reference()
 from .checks import checks
 assert len(checks)>=39
 old=T/'native_governor180_v3'
 for name in('policy.py','statistics.py','guard.py','audit.py','checks.py'):
  assert (C/name).read_text().replace(C.name,old.name)==(old/name).read_text(),name
 original=T.parent/'libero/resource_guard_v15/observer.py'
 tree=ast.parse(original.read_text());new=ast.parse((C/'sampler.py').read_text())
 functions=lambda t:{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,ast.FunctionDef)}
 a,b=functions(tree),functions(new)
 for name in('proc','sample'):assert a[name]==b[name],name
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('parse_pmon','means','aggregate')]
 namespace=dict(os=os)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(original),'exec'),namespace)
 window=json.loads((T.parent/'libero/outputs/resource_observer_v15_window.json').read_text());assert len(window)>=50
 raw=window[-1];x=aggregate(copy.deepcopy(raw),copy.deepcopy(window));y=namespace['aggregate'](copy.deepcopy(raw),copy.deepcopy(window));assert x==y
 from .sampler import replace,record_collection_error
 from .observer import read_current
 fixture=T/'outputs'/C.name/'CPU_error_fixture';fixture.mkdir(parents=True,exist_ok=False)
 initial=dict(wall_time=10.,schema='CPU_fixture_valid_sample',immutable_original_timestamp=True)
 replace(fixture/'current.json',initial);before=(fixture/'current.json').read_bytes();stamp=(fixture/'current.json').stat().st_mtime_ns
 record_collection_error(TimeoutError('CPU_injected_collection_failure'),root=fixture)
 assert(fixture/'current.json').read_bytes()==before and(fixture/'current.json').stat().st_mtime_ns==stamp
 assert read_current(fixture/'current.json',now=17.99)==initial
 for now in(18.001,30.,8.999):
  try:read_current(fixture/'current.json',now=now)
  except RuntimeError:pass
  else:raise AssertionError('Stale or future resource snapshot accepted')
 record_collection_error(ValueError('CPU_second_failure'),root=fixture)
 assert len((fixture/'errors.jsonl').read_text().splitlines())==2 and(fixture/'current.json').read_bytes()==before
 empty=fixture/'empty';empty.mkdir();record_collection_error(TimeoutError('CPU_no_prior_sample'),root=empty)
 try:read_current(empty/'current.json')
 except RuntimeError:pass
 else:raise AssertionError('No valid snapshot was accepted')
 out=T/'outputs'/C.name;out.mkdir(exist_ok=True)
 value=dict(passed=True,original_v15_proc_sample_AST_exact=True,original_policy_statistics_guard_audit_unchanged=True,actual_raw_window_statistics_identical=True,raw_sample_count=len(window),policy_checks=checks,no_CUDA_or_GPU_work=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in C.glob('*')if p.is_file()})
 path=out/'CPU_checks_v1.json';assert not path.exists();path.write_text(json.dumps(value,indent=2)+'\n');print('OWNED_OBSERVER_CPU_PASS',len(checks))
if __name__=='__main__':main()
