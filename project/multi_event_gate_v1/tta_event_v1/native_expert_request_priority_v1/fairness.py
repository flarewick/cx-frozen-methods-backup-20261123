"""Original request execution; at most six ordinary requests before one setup."""
import ast,copy,inspect,textwrap,types,heapq,queue,time,json

def install(c,output):
 service=c.service;original_call=service.call;original_process=service.process_one;execute=service.execute.__func__;burst=[0];planner_burst=[0]
 def priority(fn,default):
  setup=getattr(fn,'_cx_stage',None)=='setup'
  if setup:
   assert default==5
   row=dict(stage='setup',original_priority=5,effective_priority=1,ordinary_FIFO_sequence_unchanged=True,original_function_and_args_preserved=True,no_extra_calls_or_GPU_compute=True,global_FIFO_unchanged=False,bounded_ordinary_burst=6,function_attributes_unchanged=True)
   output.parent.mkdir(parents=True,exist_ok=True)
   with output.open('a')as f:f.write(json.dumps(row)+'\n')
   return 1
  return 0 if default==1 else default
 def take(service,timeout):
  q=service.requests;deadline=time.monotonic()+timeout
  with q.not_empty:
   while not q._qsize():
    remaining=deadline-time.monotonic()
    if remaining<=0:raise queue.Empty
    q.not_empty.wait(remaining)
   setups=[(i,x)for i,x in enumerate(q.queue)if getattr(x[3],'_cx_stage',None)=='setup']
   planners=[(i,x)for i,x in enumerate(q.queue)if getattr(x[3],'_cx_stage','').startswith('original_expert_')]
   if burst[0]>=6 and setups:
    index,item=min(setups,key=lambda row:row[1][1]);q.queue.pop(index);heapq.heapify(q.queue)
   elif planner_burst[0]>=16 and planners:
    index,item=min(planners,key=lambda row:row[1][1]);q.queue.pop(index);heapq.heapify(q.queue)
   else:item=q._get()
   q.not_full.notify()
  burst[0]=0 if getattr(item[3],'_cx_stage',None)=='setup'else burst[0]+1
  is_planner=getattr(item[3],'_cx_stage','').startswith('original_expert_')
  if is_planner:
   output.parent.mkdir(parents=True,exist_ok=True)
   with output.with_name('planner_queue_service.jsonl').open('a')as f:f.write(json.dumps(dict(stage=getattr(item[3],'_cx_stage'),effective_priority=8,requests_since_last_plan=planner_burst[0],bounded_queue_promotion_after_requests=16,original_function_and_args_preserved=True,no_extra_queries_or_compute=True))+'\n')
  planner_burst[0]=0 if is_planner else planner_burst[0]+1
  return item
 def rewrite(method,target,replacement,helper):
  node=ast.parse(textwrap.dedent(inspect.getsource(method)));before=copy.deepcopy(node);hits=[]
  for n in ast.walk(node):
   if isinstance(n,ast.Call)and ast.unparse(n)==target:hits.append(n)
  assert len(hits)==1
  class Change(ast.NodeTransformer):
   def visit_Call(self,n):
    if n is hits[0]:return copy.deepcopy(replacement)
    return self.generic_visit(n)
  changed=Change().visit(node);restored=copy.deepcopy(changed)
  class Undo(ast.NodeTransformer):
   def visit_Call(self,n):
    if isinstance(n.func,ast.Name)and n.func.id==helper:return copy.deepcopy(hits[0])
    return self.generic_visit(n)
  assert ast.dump(Undo().visit(restored))==ast.dump(before),'Only original queue expressions may change'
  ns=dict(method.__func__.__globals__,_fair18_priority=priority,_fair18_take=take)
  exec(compile(ast.fix_missing_locations(changed),__file__,'exec'),ns)
  return types.MethodType(ns[method.__name__],service)
 service.call=rewrite(original_call,"getattr(fn, '_cx_priority', 1)",ast.parse("_fair18_priority(fn,getattr(fn,'_cx_priority',1))",mode='eval').body,'_fair18_priority')
 service.process_one=rewrite(original_process,'self.requests.get(timeout=timeout)',ast.parse('_fair18_take(self,timeout=timeout)',mode='eval').body,'_fair18_take')
 assert service.execute.__func__ is execute
 return service.call
