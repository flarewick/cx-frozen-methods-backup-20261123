"""Full action/clock/score replay vs immutable old controls; explicit all-event diagnostic."""
import json,os,sys
from pathlib import Path
import numpy as np,torch
assert not torch.cuda.is_initialized()and not list(Path('/dev').glob('nvidia*'));torch.set_num_threads(1)
import benefit_followup_protocol_v2 as p,benefit_growth_runtime as runtime
runtime.install_sources()
from allgate237 import AllCandidateVeto
from benefit_event_full_audit_v134 import install_full_budget_audit
from benefit_pilot_audit import episode
import benefit_stage_audit as stage,benefit_growth_canonical_audit as canonical
from cx_tta.multi_event import EventDetector,LocalController,EventConfig
from cx_tta.empirical_onset_gate import candidate_packet
from cx_tta.structured_action_adapter import FORWARD_KEYS
from task_adapters.eval_resume import final_result
cohort=int(os.environ['CX_GATE_TRANSFER_COHORT']);old=p.HOME/('candidate_veto_cup_bowls_v200_'+str(cohort))/'dev0';reg=json.loads((p.ROOT.parent/'preregistration237.json').read_text())
for f,h in reg['controls_origin_sha256'].items():
 file=Path(f)
 if not file.exists():file=p.R/file.relative_to('/data/qiufw/cx_migration/cx_new_direction_guide_TTA')
 assert p.sha(file)==h
cp=json.loads((old/'collect_plan.json').read_text());gp=json.loads((p.ROOT/'validate_plan.json').read_text());a=json.loads((old/'collection_audit.json').read_text());model=json.loads((p.ROOT/'gate.json').read_text());proof=install_full_budget_audit();cases=[];queries=0
# Old plans sometimes contain host paths. Resolve those paths explicitly; never mutate sources.
def job_path(job):
 job=dict(job);run=Path(job['run'])
 if not run.exists():job['run']=str(p.R/run.relative_to('/data/qiufw/cx_migration/cx_new_direction_guide_TTA'))
 return job
cp['jobs']=[job_path(j)for j in cp['jobs']]
def yes(*args):return dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
for job in gp['jobs']:assert final_result(Path(job['run']),job['manifest'])['complete']
for case in a['cases']:
 task,seed=case['task'],case['seed'];gjob=next(j for j in gp['jobs']if j['task']==task);g=episode(gjob,seed);integrity=stage.audit_episode('G',g)
 b=episode(next(j for j in cp['jobs']if j['task']==task and j['branch']=='B'),seed);z=episode(next(j for j in cp['jobs']if j['task']==task and j['branch']=='A0'),seed)
 detector=EventDetector(EventConfig());controller=AllCandidateVeto(LocalController(EventConfig()),model);veto=[];scored=[]
 for q,row in enumerate(g[1]):
  with np.load(g[3]/(g[4]+'_q%03d.npz'%q),allow_pickle=False)as data:packet=candidate_packet({k:data['legal__'+k].copy()for k in FORWARD_KEYS|{'base'}},data['raw_delta'].copy())
  ev,evidence=detector.step(q,packet);assert row['multi_event']['event']==ev and row['multi_event']['evidence']==evidence
  result=controller.step(q,ev,packet,yes,enabled=True);assert all(row['multi_event'][k]==v for k,v in result.items()),(task,seed,q,'controller replay')
  assert row['candidate_veto_records']==controller.last_veto_records and row['candidate_veto_scope'].startswith('all_legal_scored_candidates')
  scored.extend(controller.last_veto_records);veto.extend(r for r in controller.last_veto_records if not r['admit']);queries+=1
 first=integrity['active_queries'][0]if integrity['active_queries']else None
 assert not canonical.exact(z,g,first)if first is not None else not canonical.exact(z,g),('A0 prefix mismatch',task,seed)
 if not veto:assert not canonical.exact(b,g),('No-veto full B mismatch',task,seed)
 # G first remains an independent comparison endpoint from the original completed report.
 previous=next(c for c in json.loads((old.parent/'final.json').read_text())['cases']if c['task']==task and c['seed']==seed)
 cases.append(dict(task=task,seed=seed,baseline_success=z[0]['success'],event_full_success=b[0]['success'],first_gate_success=previous['gate_success'],all_gate_success=g[0]['success'],B_chunks=b[0]['active_chunks'],first_G_chunks=previous['gate_active_chunks'],all_G_chunks=g[0]['active_chunks'],scored_candidates=len(scored),vetoed_candidates=len(veto),later_vetoes=sum(r['query']!=controller.first_scored_query for r in veto),baseline_prefix_exact=True,no_veto_B_exact=not veto,actions_clocks_events_and_frozen_scores_exact=True))
assert len(cases)==40
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'gate_transfer_migration_v231'));from resource231 import resource_for
resources=resource_for(p.ROOT);assert resources['passed']
p.atomic(p.ROOT.parent/'final.json',dict(passed=True,complete=True,cases=cases,actual_all_event_G=40,actual_query_controller_replays=queries,resource_audit=resources,original_full_budget_audit=proof,controls_not_modified=True,no_training=True,later_event_scores_uncalibrated_transfer=True,not_independent_confirmation_or_100_episode=True))
print('237_FINAL_PASSED',len(cases),flush=True)
