"""Strict new-scene repeatability/native-time/data audit; no efficacy claim."""
import gzip,json
from pathlib import Path
import numpy as np
from benefit_pilot_protocol import D,ROOT,TASKS,BRANCHES,sha,atomic
from cx_tta.eval_diagnostics import array_hash
from cx_tta.structured_action_adapter import ARM_INDICES
from task_adapters.eval_resume import final_result
KEYS=('observation_sha256','structured_input_sha256','rng_before_query','rng_after_feature','rng_after_sampler','base_chunk_sha256','token_sha256')
def rows(file):
 with gzip.open(file,'rt')as f:return [json.loads(line)for line in f]
def episode(job,seed):
 run=Path(job['run']);journal=json.loads((run/'journal.json').read_text());ep=next(e for e in journal['episodes']if e['seed']==seed);session=run/'sessions'/ep['process_session'];stem=f"episode{ep['episode_index']:03d}_seed{seed}"
 return ep,rows(session/'queries'/(stem+'.jsonl.gz')),rows(session/'pilot_traces'/(stem+'.jsonl.gz')),session/'pilot_traces',stem

def resource_audit():
 state=json.loads((D/'reports/target_benefit_pilot_active_20261025.json').read_text())
 path=Path(state['telemetry'].replace('/data/qiufw/cx_migration/cx_new_direction_guide_TTA',str(D.parent)))
 data=[json.loads(line)for line in path.read_text().splitlines()if line.strip()]
 release=next(r for r in reversed(data)if r.get('own_gpu_released')is True)
 checked=[r for r in data if r.get('gpu_pids')and r.get('seconds',0)>=280]
 for r in checked:
  assert r['account_gpu_count']<=3 and 4096<=r['memory_mib']<=24563
  assert min(r['mean_utilization'],r['time_weighted_mean_utilization'],r['rolling_300s_utilization'],*r['process_mean_utilization'].values(),*r['process_rolling_300s_utilization'].values())>=33
  assert all(4096<=r['process_memory_mib'].get(str(pid),r['process_memory_mib'].get(pid,0))<=24563 for pid in r['gpu_pids'])
  assert all(40<=v<=1600 for v in r['cpu'].values()) and all(v<=128*2**30 for v in r['rss_bytes'].values())
 for r in data:
  if 'memory_mib'in r:assert r['memory_mib']<=24563 and r['account_gpu_count']<=3
  if 'rss_bytes'in r:assert all(v<=128*2**30 for v in r['rss_bytes'].values())
 return dict(passed=True,telemetry=str(path),post280_samples=len(checked),release=release,startup_samples_included=True,stops=[r['stopped']for r in data if 'stopped'in r])

def main():
 plan=json.loads((ROOT/'active_plan.json').read_text());qualified=json.loads((ROOT/'qualified.json').read_text());cases=[];integrity=[];issues=[]
 for job in plan['jobs']:final_result(Path(job['run']),job['manifest'])
 for task in TASKS:
  jobs={j['branch']:j for j in plan['jobs']if j['task']==task}
  for row in qualified['tasks'][task]['attempts']:
   if not row['eligible']:continue
   seed=row['seed'];branches={b:episode(jobs[b],seed)for b in BRANCHES};aa=[];prefix=[];event_queries=[]
   for b,(ep,qr,tr,folder,stem)in branches.items():
    assert len(qr)==len(tr)==ep['queries'] and ep['complete'] and ep['passive_trace_complete']
    tick=0;native=0;active=[]
    for q,(d,t)in enumerate(zip(qr,tr)):
     assert d['query']==q==t['query'];file=folder/(stem+f'_q{q:03d}.npz')
     with np.load(file,allow_pickle=False)as archive:arrays={k:archive[k]for k in archive.files}
     assert {k:array_hash(v)for k,v in arrays.items()}==t['legal_array_sha256']
     base,action,delta=arrays['base'],arrays['actions'],arrays['applied_delta'];used=d['executed_actions'];is_active=d['multi_event']['active']
     if is_active:active.append(q)
     expected=np.zeros_like(base)
     if is_active:expected[:,ARM_INDICES]=np.clip(arrays['raw_delta'][:,ARM_INDICES],-.1,.1)*np.float32(.5)
     assert np.array_equal(delta,expected) and np.count_nonzero(delta[:,[6,13]])==0
     reconstructed=base.copy();mask=delta!=0;reconstructed[mask]=base[mask]+delta[mask]
     assert np.array_equal(action,reconstructed) and np.abs(delta).max()<=.050000001
     assert array_hash(action)==d['action_chunk_sha256'] and array_hash(base)==d['base_chunk_sha256']
     assert used==t['executed']==len(t['offline_only']['actions'])
     for k,source in [('executed_actions',action),('executed_base',base),('executed_delta',delta)]:assert np.array_equal(arrays[k],source[:used])
     assert np.array_equal(arrays['legal__base'].reshape(32,14),base)
     assert not any(any(word in k for word in ('contact','success','object_pose','offline'))for k in arrays if k.startswith('legal__'))
     for a in t['offline_only']['actions']:
      assert a['action_index']==native and a['tick_before']==tick and a['tick_after']>=tick
      # Some official no-motion commands perform zero substeps; preserve them.
      assert a['time_before_s']==tick*a['dt_s'] and a['time_after_s']==a['tick_after']*a['dt_s'] and a['dt_s']>0
      assert 'contacts'in a['offline_only']['before']and'task_success_components'in a['offline_only']['after']
      tick=a['tick_after'];native+=1
     assert t['physics_tick']==tick and t['native_count']==native
    assert native==ep['executed_actions'] and tick==ep['native_physics_ticks']
    assert len(active)==ep['active_chunks'] and len(active)==(1 if b=='B'else 0)
    integrity.append(dict(task=task,seed=seed,branch=b,queries=len(qr),executed=native,physics_ticks=tick,active_queries=active))
    if b=='B':event_queries=active
   a0,a1,b=branches['A0'],branches['A1'],branches['B']
   for field in ('success','instruction','initial_observation_signature','initial_physical_sha256','executed_actions','queries','native_physics_ticks'):
    if a0[0][field]!=a1[0][field]:aa.append(dict(field=field,query=None))
   for q in range(max(len(a0[1]),len(a1[1]))):
    if q>=min(len(a0[1]),len(a1[1])):aa.append(dict(field='query_count',query=q));break
    for key in (*KEYS,'action_chunk_sha256','applied_delta_sha256'):
     if a0[1][q][key]!=a1[1][q][key]:aa.append(dict(field=key,query=q))
    if a0[2][q]!=a1[2][q]:
     for key in ('executed','physics_tick','native_count','legal_array_sha256','offline_only'):
      if a0[2][q][key]!=a1[2][q][key]:aa.append(dict(field='passive.'+key,query=q))
   first=event_queries[0]
   for field in ('instruction','initial_observation_signature','initial_physical_sha256'):
    if a0[0][field]!=b[0][field]:prefix.append(dict(field=field,query=None))
   for q in range(first+1):
    for key in KEYS:
     if a0[1][q][key]!=b[1][q][key]:prefix.append(dict(field=key,query=q))
    if a0[2][q]['offline_only']['physical_before']!=b[2][q]['offline_only']['physical_before']:prefix.append(dict(field='physical_before',query=q))
    if q<first and a0[2][q]!=b[2][q]:prefix.append(dict(field='preintervention_full_trace',query=q))
   ok=not aa and not prefix
   classification=('both_success'if b[0]['success']else'harm')if a0[0]['success']else('rescue'if b[0]['success']else'both_fail')
   cases.append(dict(task=task,seed=seed,split=a0[0]['pilot_split'],successes={k:v[0]['success']for k,v in branches.items()},baseline_repeat_exact=not aa,baseline_repeat_differences=aa,preintervention_exact=not prefix,preintervention_differences=prefix,single_event_query=first,benefit_label=classification if ok else None))
 passed=all(c['baseline_repeat_exact']and c['preintervention_exact']for c in cases)
 if not passed:
  for case in cases:case['benefit_label']=None
 report=dict(resources=resource_audit(),complete=True,episodes=12,scenes=4,passed=passed,record_integrity_passed=True,native_execution_time_passed=True,cases=cases,episode_integrity=integrity,policy_labels_created=False,expansion_authorized=False,decision='Ready for separate larger protocol review'if passed else'STOP expansion; fix replay before benefit labels',source_sha256=plan['source_sha256'])
 atomic(D/'reports/target_benefit_pilot_acceptance_20261025.json',report)
 lines=['# 新场景12局单事件试验验收','',f"完整12局；记录/原生时序通过；重复性与干预前缀：{'通过'if passed else'未通过'}。",'', '|任务|种子|分区|A0/A1/B成功|A/A完全一致|B前缀一致|','|---|---:|---|---|---|---|']
 for c in cases:lines.append(f"|{c['task']}|{c['seed']}|{c['split']}|{'/'.join(str(int(c['successes'][b]))for b in BRANCHES)}|{c['baseline_repeat_exact']}|{c['preintervention_exact']}|")
 lines+=['','未扩大采集，未训练门控。4个场景只验收接口，不用于推断成功率改善。', '', '完整差异和每局物理tick见同名JSON。']
 (D/'reports/target_benefit_pilot_acceptance_20261025.md').write_text('\n'.join(lines)+'\n');print('PILOT_ACCEPTANCE',json.dumps(report),flush=True)
if __name__=='__main__':main()
