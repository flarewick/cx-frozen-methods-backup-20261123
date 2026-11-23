"""Original GPU AST invariant and real complete/partial final-result equivalence."""
import ast,copy,contextlib,io,json,os,tempfile
from pathlib import Path
import torch
import benefit_success_shortcircuit
import benefit_growth_runtime as growth
from benefit_growth_protocol import HOME,PRIOR,D,sha,atomic
import benefit_growth_deferred_finalize as f

def main():
 torch.set_num_threads(1);growth.bind();__import__('benefit_memory_batches').install();f.install();growth.install()
 from shared6_eval_backend import sources
 from task_adapters import eval_resume as original
 import benefit_memory_batches as batches
 batches.install()
 assert len(f.TRANSFORMS)==1 and f.TRANSFORMS[0]['GPU_body_before_finalization_AST_unchanged']
 assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'))
 with tempfile.TemporaryDirectory()as temp:
  root=Path(temp);barrier=root/'control';barrier.mkdir();flag=barrier/'memory_recycle.json';flag.write_text('{}')
  os.environ['CX_BENEFIT_RECYCLE_FILE']=str(flag);os.environ['CX_TARGET_GPU_UUID']='CPU_TEST_NO_GPU'
  outcomes=[]
  for block,task,branch in [('train0','lift_pot','A1'),('dev0','lift_pot','A0')]:
   plan=json.loads((PRIOR/block/'collect_plan.json').read_text());job=next(j for j in plan['jobs']if j['task']==task and j['branch']==branch)
   prior=json.loads((Path(job['run'])/'result.json').read_text());restore=[]
   if block=='dev0':
    # The input fixture is a real validated prefix from the closed DEV corpus.
    # The original journal checker is executed unchanged except its read source;
    # no original journal, evidence path, source, runtime or result is written.
    import inspect,textwrap
    full_reader=original.load_journal;state=full_reader(Path(job['run']),job['manifest']);assert len(state['episodes'])==20
    state=copy.deepcopy(state);state['episodes']=state['episodes'][:12];attempts=[]
    for row in state['attempts']:
     attempts.append(row)
     if row['eligible']and row['episode_index']==12:break
    state['attempts']=attempts
    node=ast.parse(textwrap.dedent(inspect.getsource(full_reader))).body[0]
    targets=[n for n in node.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='state'for t in n.targets)];assert len(targets)==1
    targets[0].value=ast.parse('copy.deepcopy(FIXTURE_STATE)',mode='eval').body
    ns=dict(full_reader.__globals__,copy=copy,FIXTURE_STATE=state);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),ns);reader=ns['load_journal']
    assert reader(Path(job['run']),job['manifest'])==state
    closure=dict(zip(original.final_result.__code__.co_freevars,[c.cell_contents for c in original.final_result.__closure__]))
    dictionaries={id(fun.__globals__):fun.__globals__ for key,fun in closure.items()if key in ('original','partial')}
    dictionaries[id(original.__dict__)]=original.__dict__
    for dictionary in dictionaries.values():
     if 'load_journal'in dictionary:restore.append((dictionary,dictionary['load_journal']));dictionary['load_journal']=reader
   try:
    expected=original.final_result(Path(job['run']),job['manifest'])
    if block=='train0':assert expected==prior and expected['complete']
    else:assert expected['completed']==12 and not expected['complete']and expected['successes']==sum(e['success']for e in state['episodes'])
    written={}
    with contextlib.redirect_stdout(io.StringIO()):loop_hash=f.execute_original_final_loop(dict(jobs=[job]),writer=lambda p,v:written.__setitem__(str(p),copy.deepcopy(v)))
    assert written[str(Path(job['run'])/'result.json')]==expected
    assert written[str(Path(job['run'])/'progress.json')]==dict(phase='finished',**expected)
    assert loop_hash==f.TRANSFORMS[0]['original_final_loop_AST_sha256']
    outcomes.append(dict(block=block,branch=branch,completed=expected['completed'],requested=expected['requested'],complete=expected['complete'],original_result_exact=True,CPU_partial_fixture_from_validated_prefix=block=='dev0'))
   finally:
    for dictionary,value in restore:dictionary['load_journal']=value
  before=original.final_result;written={}
  def fail(*args):raise RuntimeError('CPU provenance failure must propagate')
  try:
   original.final_result=fail
   try:
    with contextlib.redirect_stdout(io.StringIO()):f.execute_original_final_loop(dict(jobs=[job]),writer=lambda p,v:written.__setitem__(str(p),v))
   except RuntimeError as e:assert 'must propagate'in str(e)
   else:raise AssertionError('CPU error was suppressed')
   assert not written
  finally:original.final_result=before
  run=root/'run';session=run/'sessions/process_new';session.mkdir(parents=True);plan=dict(jobs=[dict(run=str(run),previous_sessions=[])],source_sha256=sources())
  import benefit_followup_mixed_core_v3 as mixed
  mixed.state().persist(mixed.p.ROOT/mixed.EVENTS)
  fixture=dict(all_expert_threads_finished=True,source_sha256=sources(),qualification_stream_scope_sha256=sha(mixed.scope_path()),event_journal_sha256=sha(mixed.p.ROOT/mixed.EVENTS),complete=False,expert_counts={t:r['qualified']for t,r in mixed.state().ledger.tasks.items()},expert_runtime_audits_sha256={},qualified_sha256=None,normal_memory_maintenance=True,CPU_fixture_only=True)
  atomic(barrier/'mixed_expert_checkpoint.json',fixture)
  try:f.ready_for_cpu(plan)
  except RuntimeError as e:assert 'source audit missing'in str(e)
  else:raise AssertionError('Missing original scene audit accepted')
  assert not(barrier/'cpu_finalize_intent.json').exists()
  (session/'source_runtime_audit.json').write_text('{"CPU_test_fixture":true}')
  (barrier/'mixed_expert_checkpoint.json').unlink()
  try:f.ready_for_cpu(plan)
  except RuntimeError as e:assert 'checkpoint missing' in str(e)
  else:raise AssertionError('Missing mixed checkpoint accepted')
  assert not(barrier/'cpu_finalize_intent.json').exists(), 'Failed checkpoint validation must be atomic'
  invalid=copy.deepcopy(fixture);invalid['expert_counts']['lift_pot']+=1
  atomic(barrier/'mixed_expert_checkpoint.json',invalid)
  try:f.ready_for_cpu(plan)
  except RuntimeError as e:assert 'counts differ' in str(e)
  else:raise AssertionError('Tampered expert counts accepted')
  assert not(barrier/'cpu_finalize_intent.json').exists()
  atomic(barrier/'mixed_expert_checkpoint.json',fixture)
  f.ready_for_cpu(plan);intent=json.loads((barrier/'cpu_finalize_intent.json').read_text());assert intent['worker_AST_receipt']==f.TRANSFORMS[0]
  assert intent['helper_sha256']==sha(D/'benefit_growth_deferred_finalize.py')and len(intent['source_runtime_audits_sha256'])==1
  try:f.finalize(barrier/'cpu_finalize_intent.json')
  except FileNotFoundError:pass
  else:raise AssertionError('CPU finalization accepted absent host GPU release receipt')
  (barrier/'host_gpu_release_receipt.json').write_text(json.dumps(dict(GPU_job_released=False,release=dict(own_gpu_released=False))))
  try:f.finalize(barrier/'cpu_finalize_intent.json')
  except AssertionError:pass
  else:raise AssertionError('CPU finalization accepted unreleased GPU job')
 assert not torch.cuda.is_initialized()
 proof=dict(passed=True,cpu_only=True,cuda_initialized=False,actual_worker_transform=f.TRANSFORMS[0],actual_complete_and_partial_results=outcomes,original_final_loop_AST_executed=True,no_production_result_writes=True,CPU_validation_error_propagated_before_results=True,unreleased_GPU_finalization_rejected=True,missing_GPU_phase_source_audit_rejected=True,intent_written_only_after_source_audit=True,helper_sha256=sha(D/'benefit_growth_deferred_finalize.py'),checker_sha256=sha(__file__),source_sha256=sources())
 atomic(HOME/'deferred_finalize_cpu_proof.json',proof);print('DEFERRED_CPU_PROOF_PASSED',json.dumps({k:v for k,v in proof.items()if k not in ('source_sha256',)}),flush=True)
if __name__=='__main__':main()
