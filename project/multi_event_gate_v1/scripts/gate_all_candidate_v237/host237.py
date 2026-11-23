"""Original strict host, actualGPU0/cache selection and original joint admission locks."""
import ast,fcntl,runpy,sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import entry237
import benefit_followup_protocol_v2 as p
sys.path.insert(0,str(HERE.parent/'gate_transfer_migration_v231'))
import benefit_inventory_nvml_v60 as inv;inv.install()
from gpu_policy import require_admission
from compiler_bundle import read_bundle
from shared6_native_cache_seed import Shared6NativeCacheSeed
import shared6_eval_host as cache,caches231
cfg=json.loads((p.ROOT/'migration231/target.json').read_text());bundle=p.R/cfg['bundle_relative'];meta,_=read_bundle(bundle)
assert p.sha(bundle/'bundle.json')==cfg['bundle_metadata_sha256']
native=Shared6NativeCacheSeed(p.R/cfg['native_proof_relative'],bundle);assert native.proof['source_count']==48 and not native.proof['unselected']
context=caches231.select(cfg)
def choose(uuid,training):
 assert uuid==cfg['uuid'] and Path(training).resolve()==p.TRAINING.resolve();return bundle,native,context
cache.select_cache=choose
import benefit_pilot_driver_cache
benefit_pilot_driver_cache.export=caches231.export
sys.path.insert(0,str(HERE.parent/'gate_transfer_host_locks_v227'));from locks227 import acquire,REAL_FLOCK
host=p.D/'benefit_growth_host_v3.py';main=next(n for n in ast.parse(host.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main');line=next(n.lineno for n in ast.walk(main)if isinstance(n,ast.Call)and ast.unparse(n)=="fcntl.flock(account, fcntl.LOCK_EX | fcntl.LOCK_NB)")
def flock(file,op):
 caller=sys._getframe(1)
 if op!=(fcntl.LOCK_EX|fcntl.LOCK_NB)or caller.f_code.co_filename!=str(host)or caller.f_code.co_name!='main'or caller.f_lineno!=line:return REAL_FLOCK(file,op)
 assert caller.f_locals['info']['uuid']==cfg['uuid'];acquire(file,caller.f_locals['selected'],lambda:require_admission(inv.inventory(),cfg['uuid']),lambda why,s:p.atomic(p.ROOT/'lock_wait237.json',dict(reason=why,seconds=s,no_GPU_at_callsite=True)))
fcntl.flock=flock
runpy.run_path(str(p.D/'benefit_admission_wait_host_v59.py'),run_name='__main__')
