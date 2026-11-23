"""Append-only expert membership for productive mixed GPU work, not policy outcomes.

This module is a CPU-verified building block. It is not a GPU launcher.
"""
import copy,hashlib,json,os,threading
from pathlib import Path
from benefit_growth_parallel_screen_v4 import OrderedEligibility
ALLOWED={'seed','episode_index','eligible','initial_physical_sha256','expert_plan_success','expert_success','reason','error','error_type','traceback','scene_info','deployment_screen_origin'}
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
 with temp.open('wb')as f:f.write(canonical(value)+bytes([10]));f.flush();os.fsync(f.fileno())
 temp.replace(path)

class StreamingEligibility:
 def __init__(self,starts,snapshot,protocol_sha256,expected_snapshot_sha256=None,target=20,maximum=256):
  self.starts=dict(starts);self.snapshot=copy.deepcopy(snapshot);self.snapshot_sha256=digest(snapshot);self.protocol_sha256=protocol_sha256;self.target=target;self.maximum=maximum
  if expected_snapshot_sha256 is not None and self.snapshot_sha256!=expected_snapshot_sha256:raise RuntimeError('Pinned expert prefix changed')
  if snapshot['protocol_sha256']!=protocol_sha256 or set(snapshot['tasks'])!=set(starts):raise RuntimeError('Expert scope changed')
  self.seen=set();self.events=[];self.head=self.snapshot_sha256
  for task,start in starts.items():
   rows=snapshot['tasks'][task]['attempts'];index=0
   for i,row in enumerate(rows):
    self.validate_row(task,row)
    if row['seed']!=start+i or row['episode_index']!=index:raise RuntimeError('Initial expert order changed')
    self.seen.add((task,row['seed']));index+=int(row['eligible'])
   if index!=snapshot['tasks'][task]['qualified']or index>target:raise RuntimeError('Initial qualification count changed')
  self.ledger=OrderedEligibility(starts,snapshot['tasks'],maximum,target);self.lock=self.ledger.lock
  self.initial={task:copy.deepcopy(value['attempts'])for task,value in snapshot['tasks'].items()}
  for task in starts:
   self.ledger.extra[task].update({int(k):copy.deepcopy(v)for k,v in snapshot.get('extra_completed_not_members',{}).get(task,{}).items()})
   for row in sorted(snapshot.get('pending',{}).get(task,{}).values(),key=lambda r:r['seed']):
    self.validate_row(task,row);key=(task,row['seed'])
    if key in self.seen:raise RuntimeError('Duplicate initial expert decision')
    self.seen.add(key);self.ledger.submit(task,row)
  self.assert_prefix()
 def validate_row(self,task,row):
  if task not in self.starts or set(row)-ALLOWED:raise RuntimeError('Only expert eligibility fields accepted')
  if type(row.get('seed'))is not int or not self.starts[task]<=row['seed']<self.starts[task]+self.maximum:raise RuntimeError('Expert seed outside fixed window')
  if type(row.get('eligible'))is not bool or type(row.get('episode_index'))is not int:raise RuntimeError('Malformed expert decision')
  if row['eligible']:
   value=row.get('initial_physical_sha256','')
   if row.get('expert_plan_success')is not True or row.get('expert_success')is not True or len(value)!=64 or any(c not in '0123456789abcdef'for c in value):raise RuntimeError('Eligible scene lacks original expert/physical evidence')
  canonical(row)
 def assert_prefix(self):
  if any(self.ledger.tasks[t]['attempts'][:len(rows)]!=rows for t,rows in self.initial.items()):raise RuntimeError('Retained expert prefix changed')
 def take(self,task):
  with self.lock:return self.ledger.take(task)
 def submit(self,task,row,path=None,origin='original_private_expert'):
  with self.lock:
   self.validate_row(task,row);key=(task,row['seed'])
   if key in self.seen:raise RuntimeError('Expert seed decided twice')
   event=dict(index=len(self.events),previous_hash=self.head,task=task,seed=row['seed'],row=copy.deepcopy(row),origin=origin);event['event_hash']=digest(event)
   self.ledger.submit(task,row);self.assert_prefix();self.seen.add(key);self.events.append(event);self.head=event['event_hash']
   if path is not None:self.persist(path)
   return event['event_hash']
 def persist(self,path):
  with self.lock:atomic(path,dict(schema='append_only_expert_eligibility_v3',starts=self.starts,target=self.target,maximum=self.maximum,protocol_sha256=self.protocol_sha256,initial_snapshot_sha256=self.snapshot_sha256,head=self.head,events=self.events))
 @classmethod
 def restore(cls,starts,snapshot,protocol_sha256,path,target=20,maximum=256):
  saved=json.loads(Path(path).read_text());obj=cls(starts,snapshot,protocol_sha256,saved['initial_snapshot_sha256'],target,maximum)
  if saved['starts']!=starts or saved['target']!=target or saved['maximum']!=maximum or saved['protocol_sha256']!=protocol_sha256:raise RuntimeError('Expert journal scope changed')
  for event in saved['events']:
   raw={k:v for k,v in event.items()if k!='event_hash'}
   if event['index']!=len(obj.events)or event['previous_hash']!=obj.head or event['event_hash']!=digest(raw)or event['seed']!=event['row']['seed']:raise RuntimeError('Expert journal chain changed')
   obj.submit(event['task'],event['row'],origin=event['origin'])
   if obj.head!=event['event_hash']:raise RuntimeError('Expert replay changed')
  if saved['head']!=obj.head:raise RuntimeError('Expert journal tip changed')
  return obj
 def branch_rows(self,task,branch,repeat=(0,1,6,11),validation=()):
  with self.lock:
   out=[];count=0
   for original in self.ledger.tasks[task]['attempts']:
    row=copy.deepcopy(original);index=row['episode_index'];row['qualified_index']=index
    selected=index in repeat if branch=='A1'else index in validation if branch=='G'else True
    if row['eligible']and not selected:row.update(eligible=False,reason='preregistered_branch_subset',original_expert_eligible=True)
    row['episode_index']=count;count+=int(row['eligible']);out.append(row)
   return out
 def seal(self,source_sha256):
  with self.lock:
   if not self.ledger.complete():raise RuntimeError('Cannot seal an incomplete expert cohort')
   self.assert_prefix()
   return dict(complete=True,policy_episodes=0,policy_episodes_in_this_expert_artifact=0,qualification_artifact_expert_only=True,coexecuted_with_real_policy_work=True,expert_membership_independent_of_policy_outcomes=True,protocol_sha256=self.protocol_sha256,source_sha256=source_sha256,initial_snapshot_sha256=self.snapshot_sha256,eligibility_event_chain_head=self.head,tasks=copy.deepcopy(self.ledger.tasks))

def extend_queue(queue,rows):
 """Append qualified rows without replacing any completed original episode."""
 with queue.lock:
  previous=queue.state['attempts'];length=len(previous)
  if len(rows)<length or rows[:length]!=previous:raise RuntimeError('Committed policy eligibility prefix changed')
  if len(rows)==length:return False
  completed=copy.deepcopy(queue.state['completed']);previous.extend(copy.deepcopy(rows[length:]))
  try:queue.validate()
  except BaseException:del previous[length:];raise
  if queue.state['completed']!=completed:raise RuntimeError('Completed policy results changed')
  queue._persist();queue._commit();return True
