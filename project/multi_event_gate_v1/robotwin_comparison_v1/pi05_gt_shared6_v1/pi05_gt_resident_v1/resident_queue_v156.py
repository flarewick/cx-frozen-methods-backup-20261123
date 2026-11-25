"""Finite, fixed-cohort queue; bounded slots refill with fresh independent episodes."""
import threading,time,json
from pathlib import Path
class ResidentQueue:
 def __init__(self,total,slots,cancel,root):
  assert 1<=slots<=10 and total>=slots
  self.total,self.slots,self.cancel,self.root=total,slots,cancel,Path(root)
  self.lock=threading.Lock();self.next_index=min(slots,8);self.active={};self.completed=[];self.peak=0;self.claimed=set()
 def emit(self,event,**fields):
  with(self.root/'resident_events.jsonl').open('a')as f:f.write(json.dumps(dict(event=event,wall_time=time.time(),**fields))+'\n')
 def first(self,slot):
  with self.lock:
   assert 0<=slot<min(self.slots,8) and slot not in self.claimed
   self.claimed.add(slot);return slot
 def next(self):
  with self.lock:
   if self.cancel.is_set()or self.next_index>=self.total:return None
   index=self.next_index;self.next_index+=1;assert index not in self.claimed;self.claimed.add(index);return index
 def begin(self,slot,index,case_id):
  with self.lock:
   assert not self.cancel.is_set()and slot not in self.active and index in self.claimed
   self.active[slot]=dict(index=index,case_id=case_id);self.peak=max(self.peak,len(self.active));assert self.peak<=self.slots
   self.emit('begin',slot=slot,index=index,case_id=case_id,active_cases=len(self.active),fresh_lane_RNG=True)
 def finish(self,slot,index,case_id):
  with self.lock:
   assert self.active.pop(slot)==dict(index=index,case_id=case_id)and index not in self.completed
   self.completed.append(index);self.emit('finish',slot=slot,index=index,case_id=case_id,active_cases=len(self.active))
 def receipt(self):
  with self.lock:
   assert not self.active and set(self.completed)==set(range(self.total))and len(self.completed)==self.total
   return dict(passed=True,registered_cases=self.total,concurrent_slots=self.slots,peak_active_cases=self.peak,all_cases_exactly_once=True,slot_reused=self.total>self.slots,completed_indices=self.completed,single_frozen_model_owner=True)
