"""Episode-local phase-window feedback, released once at a real caller query boundary."""
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class Ticket:
 epoch:int
 index:int
@dataclass(frozen=True)
class Feedback:
 ticket:Ticket
 source_query:int
 query:int
 observation_tick:int
 rows:tuple
 query_proof_sha256:str
class FeedbackQueue:
 def __init__(self,max_updates=3,min_lag_s=.25):
  if max_updates!=3 or min_lag_s!=.25:raise ValueError('Frozen update count/response lag')
  self.limit=max_updates;self.lag=min_lag_s;self.epoch=-1;self.reset()
 def reset(self):
  self.epoch+=1;self.events={};self.frames={};self.now=-1;self.elapsed=-math.inf;self.last_query=-1;self.released=0;self.closed=False
 def arrive(self,tick,elapsed_s,rgb_sha256):
  if self.closed or type(tick)is not int or tick<0 or not math.isfinite(elapsed_s)or len(rgb_sha256)!=64:raise ValueError('Actual arrived frame/clock required')
  if tick<self.now or elapsed_s<self.elapsed:raise ValueError('Backward/future frame ordering')
  if tick in self.frames:
   if self.frames[tick]!=(elapsed_s,rgb_sha256):raise ValueError('Same tick has conflicting RGB/physical clock')
  elif tick==self.now:raise ValueError('Conflicting arrived tick')
  self.frames[tick]=(elapsed_s,rgb_sha256);self.now=tick;self.elapsed=elapsed_s
 def register(self,source_query):
  if self.closed or self.now<0 or type(source_query)is not int or source_query<0 or source_query<self.last_query:raise ValueError('Current query registration only')
  if len(self.events)>=self.limit:raise ValueError('Frozen max3 source/update budget')
  if any(e['source_query']==source_query for e in self.events.values()):raise ValueError('Duplicate source query')
  if self.last_query==-1:self.last_query=source_query
  if source_query!=self.last_query:raise ValueError('Source is not the actual current query')
  t=Ticket(self.epoch,len(self.events));self.events[t]=dict(source_query=source_query,start_tick=self.now,start_s=self.elapsed,
   receipt=None,phase=None,complete=None,native_complete=None,used=False,windows={})
  return t
 def event(self,ticket):
  if self.closed or ticket.epoch!=self.epoch or ticket not in self.events:raise ValueError('Stale/cross-episode ticket')
  return self.events[ticket]
 def window(self,ticket,*,row,start_tick,end_tick,kind,command_start_tick,command_end_tick):
  e=self.event(ticket)
  if e['used']:return False
  if kind not in('origin','following')or type(row)is not int or row<0:raise ValueError('Known actual command segment and observation row')
  if start_tick not in self.frames or end_tick not in self.frames or end_tick!=self.now:raise ValueError('Window has not actually arrived')
  if not e['start_tick']<=start_tick<end_tick or not 25<=end_tick-start_tick<=125:raise ValueError('Fixed25..125 actual source steps')
  if not command_start_tick<=start_tick<end_tick<=command_end_tick<=self.now:raise ValueError('Borrowed future/cross-segment commands')
  if kind=='origin'and e['receipt']is not None and end_tick>e['receipt']:raise ValueError('Following commands attributed to origin')
  if kind=='following'and(e['receipt']is None or start_tick<e['receipt']):raise ValueError('Following prefix unavailable')
  if row in e['windows']:raise ValueError('Repeated feedback row')
  e['windows'][row]=(start_tick,end_tick,kind,self.frames[start_tick][0],self.frames[end_tick][0])
  return True
 def close_origin(self,ticket,*,commands_returned,complete_native32,receipt_tick,phase95_tick,phase95_elapsed_s):
  e=self.event(ticket)
  if e['receipt']is not None or receipt_tick!=self.now or receipt_tick<=e['start_tick']:raise ValueError('Actual current original receipt required once')
  if type(commands_returned)is not int or not 0<=commands_returned<=32 or type(complete_native32)is not bool:raise ValueError('Actual native32 return schema')
  if complete_native32 and commands_returned!=32:raise ValueError('Contradictory native32 receipt')
  full=bool(complete_native32 and commands_returned==32 and phase95_tick is not None)
  if full:
   if type(phase95_tick)is not int or not isinstance(phase95_elapsed_s,(int,float))or not math.isfinite(phase95_elapsed_s):raise ValueError('Actual95 physical clock required')
   if not e['start_tick']<phase95_tick<=receipt_tick or not e['start_s']<phase95_elapsed_s<=self.elapsed:raise ValueError('Future/invalid actual95')
  e.update(receipt=receipt_tick,phase=phase95_elapsed_s if full else None,complete=full,native_complete=bool(complete_native32 and commands_returned==32))
 def eligible(self,ticket):
  e=self.event(ticket)
  if not e['complete']or e['used']:return()
  return tuple(row for row,w in sorted(e['windows'].items())if w[3]>=e['phase']and w[4]-e['phase']>=self.lag and max(w[1],e['receipt'])<=self.now)
 def at_query(self,*,query,tick,query_proof_sha256):
  if self.closed or type(query)is not int or query<=self.last_query or tick!=self.now or len(query_proof_sha256)!=64:raise ValueError('One new actually observed query boundary required')
  if self.last_query>=0 and query!=self.last_query+1:raise ValueError('Skipped native query')
  if any(query>e['source_query']and not e['native_complete']for e in self.events.values()):raise ValueError('Query before original native32return or after censor')
  self.last_query=query
  for ticket,e in self.events.items():
   if query<=e['source_query']:continue
   rows=self.eligible(ticket)
   if rows:
    if self.released>=self.limit:raise ValueError('Frozen update count exceeded')
    packet=Feedback(ticket,e['source_query'],query,tick,rows,query_proof_sha256)
    e['used']=True;self.released+=1;return packet
  return None
 def finish(self):
  result=dict(released_feedback_packets=self.released,pending_sources=sum(not e['used']for e in self.events.values()),scope='Feedback packets only, no optimizer or actor updates')
  self.closed=True;return result
