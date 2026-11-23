"""Actual qualified weights, exact scientific engine,65-scene prefix and new policy negatives."""
import ast,copy,json,runpy
import torch
from .common import C,T,P,read,sha,mapped,verify,stage_folder,segment_folder
from native_policy_execution_v1.common import write

def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 for p in [*C.rglob('*.py'),*(T/'native_governor180_v2').glob('*.py')]:ast.parse(p.read_text(),filename=str(p))
 old=T/'native_calibrated_five100_v32'
 changed={'common.py','host.py','checks.py','collection_plan.py','campaign.py','cold_resume.py','cpu_admission.py','preparation.py','runtime.py','tail_resume.py','rule8_resume.py','work_checks.py','work.py','reference_order.py','AGENTS.md'}
 exact=[]
 for p in old.rglob('*'):
  if p.is_file()and p.suffix in('.py','.sh')and p.name not in changed:
   q=C/p.relative_to(old);assert q.read_text()==p.read_text().replace(old.name,C.name),str(q);exact.append(str(p.relative_to(old)))
 assert (C/'runtime.py').read_text()==(old/'runtime.py').read_text().replace(old.name,C.name);exact.append('runtime.py')
 from .reference_order import before_tail
 assert before_tail(['M_event_full','baseline','TTA','handoff_reference'])==['M_event_full','handoff_reference','TTA','baseline']
 seq=['M_event_full','baseline','TTA','zero_update','shadow_update'];seen=[]
 for branch in seq:
  seen.append(branch)
  if branch=='M_event_full':seq[:]=before_tail(seq+['handoff_reference'])
 assert seen==['M_event_full','handoff_reference','TTA','baseline','zero_update','shadow_update']
 from .work_checks import check as work_check
 assert work_check()
 assert {'runtime.py','episode.py','baseline.py','screen.py','preload.py','preflight.py','closed_audit.py','python_scheduling.py','scripts/gpu.sh','scripts/cpu.sh'}<=set(exact)
 a=ast.parse((old/'host.py').read_text().replace(old.name,C.name));b=ast.parse((C/'host.py').read_text())
 for name in('script','canonical','wait_file'):
  first=next(x for x in a.body if isinstance(x,ast.FunctionDef)and x.name==name);second=next(x for x in b.body if isinstance(x,ast.FunctionDef)and x.name==name);assert ast.dump(first)==ast.dump(second)
 from native_governor180_v2 import observer
 old_require=observer.require
 def forbidden(*a,**k):raise AssertionError('Full Rule8 check escaped CPU wait handler')
 observer.require=forbidden
 try:
  card=observer.require_admission(dict(own_gpu_count=0,cards=[dict(uuid='empty',available=True,pids=[],memory_mib=1,sm=0,own_pids=[])]),'empty');assert card['uuid']=='empty'
 finally:observer.require=old_require
 choose=next(n for n in b.body if isinstance(n,ast.FunctionDef)and n.name=='choose')
 assert any(isinstance(n,ast.Try)and "admission.require_admission(fleet, info['uuid'])"in ast.unparse(n)and "require_rule4(info['uuid'])"in ast.unparse(n)for n in ast.walk(choose))
 import native_governor180_v2.checks as policy_checks
 assert len(policy_checks.checks)>=35
 from native_governor180_v2.guard import transformed
 tree=transformed((T/'native_process_sampler_v13/strict_guard.py').read_text());source=ast.unparse(tree)
 assert 'if elapsed >= 280 and active:'not in source and 'min_cumulative_utilization != 0'in source
 assert 'steady_required=False'in source and 'return'not in source[:20]
 for token in('stop_job(process)','record_release(process, info, start, report, known)','retired(','stale_after_verified_normal_exit','max_memory_mib','Account exceeded'):
  assert token in source,token
 from native_governor180_v2.audit import diagnostic_legacy_process_windows,diagnostic_legacy_device_windows
 # Actual original telemetry fixtures exercise diagnostic source transforms without touching originals.
 import tempfile
 oldproof=read(T/'outputs/native_calibrated_official_progress_v1/qualified_065.json');assert oldproof['episodes_each_task']==65 and oldproof['official_seed']==0
 for name,digest in oldproof['input_sha256'].items():assert sha(mapped(name))==digest
 from .campaign import qualified
 folders=[stage_folder(n)for n in range(7)]+[segment_folder(n,p)for n in range(7,19)for p in(0,1)]+[segment_folder(19,0)]
 groups={};inputs={};reports=[]
 for f in folders:
  root,spec,report=qualified(f);reports.append(report)
  assert spec['official_seed']==0 and spec['candidate_checkpoint_sha256']=='78f1849a0f825398a6aa43cc3eec9cb09a814fc472a495179ac97379f5a044d5'
  assert 'OIDN Error:'not in(root/'GPU_host.log').read_text(errors='replace')
  for p in(f/'actual_host_wait_exit.json',root/'terminal_audit.json',root/'protocol.json',f/'pilot_complete.json'):inputs[str(p)]=sha(p)
 for task in reports[0]['groups']:
  rows=[r for v in reports for r in v['groups'][task]['records']];assert len(rows)==98 and len({r['case_id']for r in rows})==98 and sorted(r['episode_index']for r in rows)==list(range(98))
 final=mapped(read(segment_folder(19,0)/'execution_state.json')['root']);state=read(final/'screening_complete.json')['state'];assert all(v['accepted']==98 for v in state.values())
 assert segment_folder(12,1).is_relative_to(T/'outputs/native_calibrated_five100_v25')and segment_folder(13,0).is_relative_to(T/'outputs/native_calibrated_five100_v26')and segment_folder(13,1).is_relative_to(T/'outputs/native_calibrated_five100_v27')and segment_folder(16,0).is_relative_to(T/'outputs/native_calibrated_five100_v27')and segment_folder(16,1).is_relative_to(T/'outputs/native_calibrated_five100_v28')and segment_folder(17,0).is_relative_to(T/'outputs/native_calibrated_five100_v28')and segment_folder(17,1).is_relative_to(T/'outputs/native_calibrated_five100_v29')and segment_folder(18,0).is_relative_to(T/'outputs/native_calibrated_five100_v29')and segment_folder(18,1).is_relative_to(T/'outputs/native_calibrated_five100_v30')and segment_folder(19,0).is_relative_to(T/'outputs/native_calibrated_five100_v32')and segment_folder(19,1).is_relative_to(T/'outputs/native_calibrated_five100_v33')
 function=next(n for n in ast.parse((C/'collection_plan.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='freeze_stage')
 loop=next(n for n in function.body[1].body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='proof')
 exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(C/'collection_plan.py'),'exec'),dict(T=T,read=read,verify=verify,sha=sha,value={'input_sha256':{}}))
 from .rule8_resume import evidence
 total,bindings,failed=evidence();assert total==5169.15 and failed['inherited_stage_GPU_seconds']==4497.429
 from .preparation import collect
 template=dict(failed);template['inherited_stage_GPU_seconds']=total
 cached=collect(template,0,template['initial_screen_state']);assert cached['cached_valid_cases']==9 and set(cached['task_cache'])==set(failed['tasks'])
 assert not cached['primary_policy_results_imported']and not cached['old_masks_or_teacher_outputs_imported']
 inputs.update(bindings);inputs.update(cached['input_sha256'])
 from .collection_plan import confirmed_candidate
 from native_visible_response_bridge_v1.bridge import load_model
 reportfile,qualification,q=confirmed_candidate();model,trained=load_model(qualification)
 assert trained['actual_updates']==1000 and trained['minimum_distinct_FIT_cases']==7 and q['training_checkpoint_sha256']==spec['candidate_checkpoint_sha256']
 from native_calibrated_official_authorization_v1.authority import AUTH,validate
 auth=read(AUTH);report=read(reportfile);assert validate(auth,report,q['training_checkpoint_sha256'])
 for v in(None,dict(auth,explicit_user_authorization=False),dict(auth,official_seed=1)):
  try:validate(v,report,q['training_checkpoint_sha256'])
  except(AssertionError,ValueError):pass
  else:raise AssertionError('Official authorization negative control failed')
 assert report['paired_net_success_change']==-1
 assert not torch.cuda.is_initialized()and not list((T/'outputs'/C.name).glob('official_*/batches/*/runs/*'))
 inputs.update({str(p):sha(p)for p in(qualification,reportfile,AUTH,T/'outputs/native_calibrated_official_progress_v1/qualified_065.json')})
 for proof in(T/'outputs/native_governor180_v1/CPU_actual_raw_replay_v1/CPU_governor180_resource_audit.json',T/'outputs/native_governor180_v1/CPU_legacy_diagnostic_replay_v1/CPU_idle_missing_lower_bound_audit.json',T/'outputs/native_governor180_v1/CPU_legacy_diagnostic_replay_v1/device_diagnostic.json'):
  assert read(proof)['passed'];inputs[str(proof)]=sha(proof)
 source={str(p):sha(p)for p in C.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md')}
 source.update({str(p):sha(p)for p in(T/'native_governor180_v2').glob('*')if p.is_file()})
 write(T/'outputs'/C.name/'CPU_contract_checks_v1.json',dict(passed=True,policy_negative_checks=policy_checks.checks,exact_scientific_files=exact,original98_unique_scenes_each_task_requalified=True,current_actual_fixed1000_min7_model_loaded=True,old_DEV_CONF_not_relabelled=True,no_GPU_or_optimizer_or_new_EVAL=True,source_sha256=source,input_sha256=inputs))
 print('OFFICIAL29_TYPED_MODEL_ORIGINAL98_RULE8_FINAL_PURE9_PASS',flush=True)
if __name__=='__main__':main()
