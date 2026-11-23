"""Return frozen all-candidate scope to original certifiedGPU3, explicit oldGPU0 origin."""
import os,sys,inspect
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'gate_all_candidate_v237'));import entry237
import benefit_growth_runtime as runtime
prior=runtime.install_sources

def sources():
 prior()
 import shared6_eval_backend as b
 if getattr(b,'_original_GPU3_241',False):return
 # Undo only the earlier physicalGPU0 transport wrapper; exact original functions return.
 check=b.check_bundle;restore=b.restore_cache
 cv=inspect.getclosurevars(check).nonlocals;rv=inspect.getclosurevars(restore).nonlocals
 assert check.__module__=='transport231'and restore.__module__=='transport231'
 b.check_bundle=cv['original_check'];b.restore_cache=rv['original_restore']
 assert b.check_bundle.__module__==b.restore_cache.__module__=='shared6_eval_backend'
 for name,module in tuple(sys.modules.items()):
  if module is not None and name.startswith(('shared6_','benefit_','model_'))and getattr(module,'restore_cache',None)is restore:setattr(module,'restore_cache',b.restore_cache)
 b._original_GPU3_241=True
runtime.install_sources=sources
import bridge241;bridge241.install()
if os.environ.get('CX_BENEFIT_FOLLOWUP_CONTAINER')!='1':
 import user_container as c,resource_entry
 from evaluate import CANONICAL_PROJECT,add_env
 resource_entry.WrappedPopen.__init__.__globals__['_startup211_entry']=HERE/'entry241.py'
 before=c.command;old=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_all_candidate_v237/entry237.py');new=str(CANONICAL_PROJECT/'multi_event_gate_v1/scripts/gate_all_candidate_migration_v241/entry241.py')
 def command(*a,**kw):
  cmd=before(*a,**kw)
  for i,v in enumerate(cmd):
   if v==old:cmd[i]=new
   elif isinstance(v,str)and "__import__('entry237');"in v:cmd[i]="import sys;sys.path.insert(0,"+repr(str(Path(new).parent))+");__import__('entry241');"+v
  return cmd
 c.command=command
if __name__=='__main__':__import__('resource_entry').entry.execute(Path(sys.argv[1]),sys.argv[2:])
