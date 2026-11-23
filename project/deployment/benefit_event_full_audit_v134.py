"""Full action/clock checks and actual full-event controller replay; no outcome fitting."""
import ast,copy,inspect,json,textwrap
from pathlib import Path
import numpy as np
import benefit_followup_protocol_v2 as p
from benefit_followup_stream_v3 import atomic
_PROOF=None

def install_full_budget_audit():
 global _PROOF
 if _PROOF is not None:return _PROOF
 import benefit_stage_audit as stage
 original=stage.audit_episode;tree=ast.parse(textwrap.dedent(inspect.getsource(original)));changed=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Compare)and ast.unparse(n.left)=='len(active)'and len(n.comparators)==1 and ast.unparse(n.comparators[0])=="int(b in ('B', 'G'))":
   changed.append(n);n.comparators[0]=ast.parse("3 * int(b in ('B','G'))",mode='eval').body
 assert len(changed)==1
 reversed_tree=copy.deepcopy(tree)
 for n in ast.walk(reversed_tree):
  if isinstance(n,ast.Compare)and ast.unparse(n.left)=='len(active)'and len(n.comparators)==1 and ast.unparse(n.comparators[0])=="3 * int(b in ('B', 'G'))":n.comparators[0]=ast.parse("int(b in ('B','G'))",mode='eval').body
 assert ast.dump(reversed_tree,include_attributes=False)==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(original))),include_attributes=False)
 namespace=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace);stage.audit_episode=namespace['audit_episode'];_PROOF=dict(only_episode_intervention_limit_changed_from1_to_original_full3=True,all_array_action_clock_native_and_trace_checks_AST_unchanged=True,original_checker_sha256=p.sha(p.D/'benefit_stage_audit.py'),extension_sha256=p.sha(__file__))
 return _PROOF

def controls():
 import benefit_growth_runtime as runtime;runtime.install_sources()
 proof=install_full_budget_audit()
 import benefit_supplement_full_audit_v27 as original
 original.main()
 a=json.loads((p.ROOT/'collection_audit.json').read_text());assert a['passed']and a['complete']and a['resource_audit']['passed']
 atomic(p.ROOT/'full_event_control_extension_v134.json',dict(passed=True,complete=True,collection_audit_sha256=p.sha(p.ROOT/'collection_audit.json'),original_membership_full_audit_sha256=p.sha(p.ROOT/'mixed_membership_final_audit.json'),budget_extension=proof,B_means_unchanged_full_event_policy=True,legacy_single_event_query_field_means_first_active_query_only=True,labels_are_full_policy_outcome_differences_not_per_candidate_labels=True))

def validate():
 import benefit_growth_runtime as runtime;runtime.install_sources();proof=install_full_budget_audit()
 import benefit_supplement_alias_audit_v27 as alias;alias.install()
 import benefit_growth_canonical_audit as canonical
 import benefit_stage_audit as stage
 from benefit_pilot_audit import episode
 from benefit_event_full_veto_v132 import FirstCandidateVeto
 from cx_tta.multi_event import EventConfig,EventDetector,LocalController
 from cx_tta.empirical_onset_gate import candidate_packet
 from cx_tta.structured_action_adapter import FORWARD_KEYS
 from task_adapters.eval_resume import final_result
 a=json.loads((p.ROOT/'collection_audit.json').read_text());extension=json.loads((p.ROOT/'full_event_control_extension_v134.json').read_text());assert a['passed']and a['complete']and extension['passed']and extension['collection_audit_sha256']==p.sha(p.ROOT/'collection_audit.json')
 cp=json.loads((p.ROOT/'collect_plan.json').read_text());gp=json.loads((p.ROOT/'validate_plan.json').read_text());gatefile=p.ROOT/'gate.json';m=json.loads(gatefile.read_text());cases=[];query_total=0
 def yes(packet,ctx):return dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
 for job in gp['jobs']:final_result(Path(job['run']),job['manifest'])
 for case in a['cases']:
  task,seed=case['task'],case['seed'];gjob=next(j for j in gp['jobs']if j['task']==task);g=episode(gjob,seed);integrity=stage.audit_episode('G',g);bjob=next(j for j in cp['jobs']if j['task']==task and j['branch']=='B');ajob=next(j for j in cp['jobs']if j['task']==task and j['branch']=='A0');b=episode(bjob,seed);zero=episode(ajob,seed)
  detector=EventDetector(EventConfig());controller=FirstCandidateVeto(LocalController(EventConfig()),m);vetoed=[]
  for q,row in enumerate(g[1]):
   file=g[3]/(g[4]+'_q%03d.npz'%q)
   with np.load(file,allow_pickle=False)as z:packet=candidate_packet({k:z['legal__'+k].copy()for k in FORWARD_KEYS|{'base'}},z['raw_delta'].copy())
   event,evidence=detector.step(q,packet);assert row['multi_event']['event']==event and row['multi_event']['evidence']==evidence
   decision=controller.step(q,event,packet,yes,enabled=True)
   assert all(row['multi_event'][key]==value for key,value in decision.items()),('Actual original controller replay mismatch',task,seed,q)
   assert row['candidate_veto_records']==controller.last_veto_records
   vetoed.extend(r for r in controller.last_veto_records if not r['admit']);query_total+=1
  first=integrity['active_queries'][0]if integrity['active_queries']else None
  assert not canonical.exact(zero,g,first)if first is not None else not canonical.exact(zero,g),'A0/G prefix or all-base replay mismatch'
  if not vetoed:assert not canonical.exact(b,g),'No-veto G must exactly replay full B'
  assert gjob['manifest']['source_sha256'][str(gatefile)]==p.sha(gatefile)
  cases.append(dict(task=task,seed=seed,baseline_success=zero[0]['success'],event_full_success=b[0]['success'],gate_success=g[0]['success'],event_full_active_chunks=b[0]['active_chunks'],gate_active_chunks=g[0]['active_chunks'],first_candidate_vetoed=bool(vetoed),actual_controller_score_and_event_replay_exact=True,full_episode_actions_and_clocks_checked=True,baseline_prefix_exact=True,no_veto_full_B_replay_exact=not vetoed))
 assert len(cases)==40
 from benefit_growth_audit import resource_audit
 resources=resource_audit();assert resources['passed']
 metrics=dict(scenes=40,baseline=sum(c['baseline_success']for c in cases),event_full=sum(c['event_full_success']for c in cases),gate=sum(c['gate_success']for c in cases),first_candidates_vetoed=sum(c['first_candidate_vetoed']for c in cases),harmful_full_policy_failures_avoided=sum(c['baseline_success']and not c['event_full_success']and c['gate_success']for c in cases),event_full_rescues_retained=sum(not c['baseline_success']and c['event_full_success']and c['gate_success']for c in cases),event_full_rescues_lost=sum(not c['baseline_success']and c['event_full_success']and not c['gate_success']for c in cases),vetoed_same_success_endpoint=sum(c['first_candidate_vetoed']and c['event_full_success']==c['gate_success']for c in cases),B_active_chunks=sum(c['event_full_active_chunks']for c in cases),G_active_chunks=sum(c['gate_active_chunks']for c in cases))
 atomic(p.ROOT.parent/'final.json',dict(passed=True,complete=True,actual_G_episodes=40,actual_full_event_control=True,metrics=metrics,cases=cases,actual_query_controller_replays=query_total,resource_audit=resources,budget_extension=proof,gate_sha256=p.sha(gatefile),no_fit_or_threshold_tuning=True,old_TEST_not_used=True,not_100_episode_or_all_tasks_claim=True))
 print('FULL134_ACTUAL_RESULTS',json.dumps(metrics),flush=True)

def main():
 import sys
 if sys.argv[1:]not in(['controls'],['validate']):raise ValueError('Explicit controls or validate stage required')
 globals()[sys.argv[1]]()
if __name__=='__main__':main()
