"""Sequential qualified stages per group; no automatic failed-GPU retries."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from .common import T,P,C,read,sha
from .qualification import qualified_stage
from .configuration import GROUPS,bounds
from .adoption import folder as stage_folder

def main():
 a=argparse.ArgumentParser();a.add_argument('--group',choices=tuple(GROUPS),required=True);args=a.parse_args();base=T/'outputs'/C.name;folder=base/('supervision_'+args.group);folder.mkdir(parents=True,exist_ok=True)
 with(folder/'supervision.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for number in range(17):
   stage=args.group+'_'+str(number).zfill(3);where=stage_folder(args.group,number)
   if(where/'actual_host_wait_exit.json').exists():qualified_stage(where);continue
   assert not(where/'execution_state.json').exists(),'Inspect interrupted batch; no unchanged GPU restart'
   assert not(where/'supervision_actual_exit.json').exists()
   env=dict(os.environ,CX_ONPOLICY_STAGE=stage,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(P/'deployment')+':'+str(T))
   where.mkdir(exist_ok=True)
   with(where/'supervision_host.log').open('x')as log:
    p=subprocess.Popen([sys.executable,'-B','-u','-m',C.name+'.launch'],env=env,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
   (where/'supervision_actual_exit.json').write_text(json.dumps(dict(exit_code=code,pid=p.pid,actual_Popen_wait_return=True,proc_absent_after_wait=not Path('/proc',str(p.pid)).exists()))+'\n');assert code==0
   qualified_stage(where)
   aggregate(args.group)
  value=aggregate(args.group);assert value['scenes_per_task']==100
  path=folder/'official100_complete.json';assert not path.exists();path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def aggregate(group):
 base=T/'outputs'/C.name;rows=[];inputs={};qualified=0;cfg=GROUPS[group]
 for i in range(17):
  where=stage_folder(group,i)
  if not(where/'actual_host_wait_exit.json').exists():break
  root,spec=qualified_stage(where);start,end=bounds(i);r=read(root/'CPU_episode_audit.json')['records']
  assert spec['group']==group and spec['episodes_per_task']==end and len(r)==len(cfg['tasks'])*(end-start)
  for task in cfg['tasks']:assert sorted(x['episode_index']for x in r if x['task']==task)==list(range(start,end))
  rows.extend(r);qualified=end
  for p in(where/'actual_host_wait_exit.json',root/'CPU_episode_audit.json',root/'terminal_audit.json'):inputs[str(p)]=sha(p)
 assert len({r['case_id']for r in rows})==len(rows)
 groups={}
 for task in cfg['tasks']:
  rr=[r for r in rows if r['task']==task];assert len(rr)==qualified
  groups[task]=dict(scenes=len(rr),control_successes=sum(r['full']for r in rr),TTA_successes=sum(r['TTA']for r in rr),rescues=sum(r['rescue']for r in rr),harms=sum(r['harm']for r in rr),accepted_update_scenes=sum(r['branches']['TTA']['accepted_updates']>0 for r in rr),changed_action_scenes=sum(r['first_changed_query']is not None for r in rr))
 value=dict(complete=qualified==100,group=group,official_seed=0,scenes_per_task=qualified,groups=groups,input_sha256=inputs,no_failed_or_partial_branches_promoted=True)
 if qualified:
  path=base/('supervision_'+group)/('qualified_'+str(qualified).zfill(3)+'.json')
  if path.exists():assert read(path)==value
  else:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 return value
if __name__=='__main__':main()
