"""Tasknames used only for simulation ownership, never predictor inputs."""
import queue
TASKS=('stack_blocks_two','stack_bowls_two')
def assigned_task(lane):
 i=int(lane)
 if not 0<=i<4:raise ValueError('Exactly four CPU lanes')
 return TASKS[i%2]
def make_queues():return {task:queue.Queue() for task in TASKS}
def enqueue(queues,scenes,index):queues[scenes[index]['task']].put(index)
def checks():
 q=make_queues();scenes=[dict(task=t) for _ in range(5) for t in TASKS]
 for i in range(len(scenes)):enqueue(q,scenes,i)
 assert [sum(assigned_task(str(i))==t for i in range(4)) for t in TASKS]==[2,2]
 for t in TASKS:
  seen=[]
  while not q[t].empty():seen.append(q[t].get())
  assert len(seen)==5 and all(scenes[i]['task']==t for i in seen)
 from .throughput import TaskPool
 p=TaskPool();created=[]
 def factory():created.append(1);return object()
 a=p.get(TASKS[0],factory);assert p.get(TASKS[0],factory)is a and len(created)==1
 rejected=False
 try:p.get(TASKS[1],factory)
 except ValueError:rejected=True
 assert rejected and len(p.tasks)==1
 return dict(four_fixed_lanes_two_per_task_queue_ownership=True,one_task_cache_per_lane_rejects_second_type=True)
