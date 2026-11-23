"""Only CPU scheduling/receipt edits; original inference body remains intact."""
def dispatcher_edits():
 return[
 ("workers=[];logs=[];busy=[None]*4;assigned=set();began=time.monotonic()","workers=[];logs=[];busy=[None]*4;budgets=[0]*4;assigned=set();began=time.monotonic()"),
 ("if key is not None and(root/'answers'/key).exists():busy[n]=None","if key is not None and(root/'answers'/key).exists():busy[n]=None;budgets[n]=0"),
 ("threads_per_worker=2)","threads_per_worker='adaptive_2_4_8',maximum_active_CPU_threads=8)"),
 ("""   for req in pending:
    if None not in busy:break
    n=busy.index(None);row=read(req);assert set(row)=={'frame_file','frame_sha256','instruction','content_key'}
    write(root/'workers'/str(n)/'assigned'/req.name,row);busy[n]=req.name;assigned.add(req.name)""","""   free_at_start=busy.count(None)
   from native_semantic_budget_pool_v1.pool import allocate
   for position,req in enumerate(pending):
    if None not in busy:break
    threads=allocate(8-sum(budgets),len(pending)-position,free_at_start)
    if not threads:break
    n=busy.index(None);row=read(req);assert set(row)=={'frame_file','frame_sha256','instruction','content_key'}
    lease=dict(content_key=row['content_key'],worker=n,threads=threads,active_before=list(budgets),maximum_active_CPU_threads=8)
    lease['active_after']=sum(budgets)+threads;assert lease['active_after']<=8
    write(root/'workers'/str(n)/'controls'/req.name,lease)
    write(root/'CPU_thread_budget'/req.name,lease)
    write(root/'workers'/str(n)/'assigned'/req.name,row);busy[n]=req.name;budgets[n]=threads;assigned.add(req.name)"""),
 ("independent_CPU_workers=4,actual_worker_exit_codes=","independent_CPU_workers=4,maximum_active_CPU_threads=8,CPU_thread_budget_guarded=True,actual_worker_exit_codes=")]

def worker_edits():
 return[
 ("   before=rng_digest();cpu=torch.random.get_rng_state().clone();start=time.monotonic()","""   allocation=work/'controls'/req.name;lease=read(allocation);threads=lease['threads']
   assert lease['content_key']==row['content_key']and lease['worker']==a.worker and threads in(2,4,8)and lease['maximum_active_CPU_threads']==8 and lease['active_after']<=8
   torch.set_num_threads(threads);assert torch.get_num_threads()==threads
   before=rng_digest();cpu=torch.random.get_rng_state().clone();start=time.monotonic()"""),
 ("actual_CPU_inference=inferred,cold_CPU_inference_seconds=","actual_CPU_inference=inferred,actual_CPU_threads=threads,CPU_thread_allocation_file=str(allocation),CPU_thread_allocation_sha256=sha(allocation),cold_CPU_inference_seconds=")]

def transform(text,kind):
 original=text
 for old,new in(dispatcher_edits()if kind=='dispatcher'else worker_edits()):
  assert text.count(old)==1,('Unexpected semantic source',kind,old[:60]);text=text.replace(old,new)
 assert restore(text,kind)==original
 return text

def restore(text,kind):
 for old,new in reversed(dispatcher_edits()if kind=='dispatcher'else worker_edits()):
  assert text.count(new)==1,('Unexpected adaptive semantic source',kind,new[:60]);text=text.replace(new,old)
 return text
