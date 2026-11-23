"""Whole-episode integrity, exact paired replay and split-separated benefits."""
import json,gzip
from pathlib import Path
import numpy as np
from benefit_stage_protocol import D,ROOT,TASKS,sha,atomic,split,selected
from benefit_pilot_audit import rows,episode,KEYS
from cx_tta.eval_diagnostics import array_hash
from cx_tta.structured_action_adapter import ARM_INDICES,FORWARD_KEYS
from task_adapters.eval_resume import final_result

def audit_episode(b,item):
 ep,qr,tr,folder,stem=item
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
 assert len(active)==ep['active_chunks'] and len(active)<=int(b in ('B','G'))
 return dict(queries=len(qr),executed=native,physics_ticks=tick,active_queries=active)

def exact(a,b,limit=None):
 issues=[]
 fields=('instruction','initial_observation_signature','initial_physical_sha256')
 if limit is None:fields+=('success','executed_actions','queries','native_physics_ticks')
 for f in fields:
  if a[0][f]!=b[0][f]:issues.append((f,None))
 count=max(len(a[1]),len(b[1]))if limit is None else limit+1
 for q in range(count):
  if q>=min(len(a[1]),len(b[1])):issues.append(('query_count',q));break
  keys=(*KEYS,'action_chunk_sha256','applied_delta_sha256')if limit is None or q<limit else KEYS
  for k in keys:
   if a[1][q][k]!=b[1][q][k]:issues.append((k,q))
  if limit is None or q<limit:
   if a[2][q]!=b[2][q]:issues.append(('full_passive_trace',q))
  elif a[2][q]['offline_only']['physical_before']!=b[2][q]['offline_only']['physical_before']:issues.append(('physical_before',q))
 return issues

def resource_audit():
 files=sorted((D/'reports').glob('target_benefit_stage_gpu*_*.jsonl'))
 checked=0;releases=[];early_stopped=[]
 for p in files:
  data=[json.loads(x)for x in p.read_text().splitlines()if x.strip()]
  release=next((r for r in reversed(data)if r.get('own_gpu_released')is True),None)
  assert release is not None,('GPU not released',str(p));releases.append(dict(path=str(p),release=release))
  stop=[r['stopped']for r in data if 'stopped'in r]
  bounded_stop=bool(stop)and release['cleanup_seconds_after_start']<300
  if bounded_stop:early_stopped.append(dict(path=str(p),stops=stop,release=release))
  for r in data:
   if 'account_gpu_count'in r:assert r['account_gpu_count']<=3
   if 'memory_mib'in r:assert r['memory_mib']<=24563
   if 'rss_bytes'in r:assert all(v<=128*2**30 for v in r['rss_bytes'].values())
   if r.get('gpu_pids')and r.get('seconds',0)>=(300 if bounded_stop else 280):
    checked+=1
    assert 4096<=r['memory_mib']<=24563
    assert min(r['mean_utilization'],r['time_weighted_mean_utilization'],r['rolling_300s_utilization'],*r['process_mean_utilization'].values(),*r['process_rolling_300s_utilization'].values())>=33
    assert all(40<=v<=1600 for v in r['cpu'].values())
    assert all(4096<=v<=24563 for v in r['process_memory_mib'].values())
 return dict(passed=True,post280_samples=checked,sessions=releases,early_stops_released_before300=early_stopped)

def collect():
 plan=json.loads((ROOT/'collect_plan.json').read_text());q=json.loads((ROOT/'qualified.json').read_text());cases=[];integrity=[]
 for n,h in plan['source_sha256'].items():assert sha(D/n)==h
 for j in plan['jobs']:final_result(Path(j['run']),j['manifest'])
 for task in TASKS:
  jobs={j['branch']:j for j in plan['jobs']if j['task']==task}
  for r in q['tasks'][task]['attempts']:
   if not r['eligible']:continue
   index=r['episode_index'];seed=r['seed'];bs=('A0','A1','B')if selected('A1',index)else('A0','B');items={b:episode(jobs[b],seed)for b in bs};proof={}
   for b,item in items.items():
    proof[b]=audit_episode(b,item);integrity.append(dict(task=task,seed=seed,branch=b,**proof[b]))
   aa=exact(items['A0'],items['A1'])if 'A1'in items else[]
   active=proof['B']['active_queries'];first=active[0]if active else None
   prefix=exact(items['A0'],items['B'],first)if first is not None else exact(items['A0'],items['B'])
   a,b=items['A0'][0]['success'],items['B'][0]['success'];label=('both_success'if b else'harm')if a else('rescue'if b else'both_fail')
   cases.append(dict(task=task,seed=seed,qualified_index=index,split=split(index),successes={k:v[0]['success']for k,v in items.items()},aa_checked='A1'in items,aa_differences=aa,preintervention_differences=prefix,single_event_query=first,benefit_label=label if not aa and not prefix and first is not None else None))
 assert len(cases)==40 and len(integrity)==88
 passed=all(not c['aa_differences']and not c['preintervention_differences']for c in cases)
 if not passed:
  for c in cases:c['benefit_label']=None
 counts={s:{k:sum(c['benefit_label']==k and c['split']==s for c in cases)for k in ('both_fail','rescue','harm','both_success')}for s in ('TRAIN','VAL')}
 by_task={task:{s:{k:sum(c['task']==task and c['split']==s and c['benefit_label']==k for c in cases)for k in ('both_fail','rescue','harm','both_success')}for s in ('TRAIN','VAL')}for task in TASKS}
 paired={task:{s:dict(scenes=sum(c['task']==task and c['split']==s for c in cases),baseline_successes=sum(c['successes']['A0'] for c in cases if c['task']==task and c['split']==s),single_event_successes=sum(c['successes']['B'] for c in cases if c['task']==task and c['split']==s),no_event_scenes=sum(c['single_event_query']is None for c in cases if c['task']==task and c['split']==s))for s in ('TRAIN','VAL')}for task in TASKS}
 report=dict(passed=passed,complete=True,scenes=40,episodes=88,aa_scenes=8,cases=cases,episode_integrity=integrity,distribution=counts,distribution_by_task=by_task,paired_successes=paired,resource_audit=resource_audit(),source_sha256=plan['source_sha256'],protocol_sha256=sha(ROOT/'protocol.json'),qualified_sha256=sha(ROOT/'qualified.json'),EVAL_labels_used=False)
 atomic(ROOT/'collection_audit.json',report);atomic(D/'reports/target_benefit_stage_distribution_20261026.json',report)
 if not passed:raise RuntimeError('Paired replay failed; benefit labels blocked')
 lines=['# 新TRAIN/VAL单事件干预收益分布','',f'完整40场景、88局；A/A及干预前缀检查通过：{passed}。','', '|任务|分区|场景|A0成功|B成功|救回|损害|均失败|均成功|无事件|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for task in TASKS:
  for split_name in ('TRAIN','VAL'):
   a=paired[task][split_name];b=by_task[task][split_name]
   lines.append(f"|{task}|{split_name}|{a['scenes']}|{a['baseline_successes']}|{a['single_event_successes']}|{b['rescue']}|{b['harm']}|{b['both_fail']}|{b['both_success']}|{a['no_event_scenes']}|")
 lines+=['','标签只针对首次事件的A0/B配对效果；后续事件不参与。分区和模型参数均已预先冻结，VAL不参与拟合或调阈值。小样本不能证明正式评估改善。']
 (D/'reports/target_benefit_stage_distribution_20261026.md').write_text('\n'.join(lines)+'\n')
 print('BENEFIT_STAGE_COLLECTION_AUDITED',json.dumps(dict(scenes=40,episodes=88,distribution=counts)),flush=True)
 return report

def training_rows(report):
 plan=json.loads((ROOT/'collect_plan.json').read_text());out=[]
 from benefit_stage_gate import features
 for c in report['cases']:
  if c['benefit_label']is None:continue
  job=next(j for j in plan['jobs']if j['task']==c['task']and j['branch']=='A0');item=episode(job,c['seed']);first=c['single_event_query'];ep,qr,tr,folder,stem=item
  with np.load(folder/(stem+f'_q{first:03d}.npz'),allow_pickle=False)as z:
   raw=z['raw_delta'];delta=np.zeros((1,32,14),np.float32);delta[0][:,ARM_INDICES]=raw[:,ARM_INDICES]
   packet={k:z['legal__'+k].copy()for k in FORWARD_KEYS|{'base'}};packet['candidate_delta']=delta
  records=qr[first]['multi_event']['records'];context=next(r['context']for r in records if r.get('reason')=='new_event_scored')
  out.append(dict(task=c['task'],seed=c['seed'],split=c['split'],label=c['benefit_label'],x=features(packet,context).tolist()))
 return out

def train():
 import torch
 torch.set_num_threads(1)
 from benefit_stage_gate import CLASSES
 report=json.loads((ROOT/'collection_audit.json').read_text());assert report['passed']
 data=training_rows(report);train=[r for r in data if r['split']=='TRAIN'];assert train
 x=np.asarray([r['x']for r in train],np.float64);y=np.asarray([CLASSES.index(r['label'])for r in train],np.int64)
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
 xx=torch.from_numpy((x-mean)/scale);yy=torch.from_numpy(y)
 weight=torch.zeros((x.shape[1],4),dtype=torch.float64,requires_grad=True);bias=torch.zeros(4,dtype=torch.float64,requires_grad=True)
 opt=torch.optim.LBFGS([weight,bias],lr=1.,max_iter=100,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
 def closure():
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(xx@weight+bias,yy)+.5*(weight.square().sum()+bias.square().sum())
  loss.backward();return loss
 initial=float(closure().detach());opt.step(closure);final=float(closure().detach())
 assert final<=initial+1e-10 and np.isfinite(final)and not torch.cuda.is_initialized()
 model=dict(schema='first_event_gate_v1',classes=list(CLASSES),mean=mean.tolist(),scale=scale.tolist(),weight=weight.detach().numpy().tolist(),bias=bias.detach().numpy().tolist(),fail_closed_no_train_rescue=not np.any(y==1),train_scenes=len(train),train_class_counts={k:sum(r['label']==k for r in train)for k in CLASSES},initial_loss=initial,final_loss=final,collection_audit_sha256=sha(ROOT/'collection_audit.json'),protocol_sha256=sha(ROOT/'protocol.json'),feature_source_sha256=sha(D/'benefit_stage_gate.py'),training_source_sha256=sha(__file__),CPU_only=True,validation_used_for_fit=False)
 if (ROOT/'gate.json').exists():assert json.loads((ROOT/'gate.json').read_text())==model,'Gate already frozen with different bytes'
 else:atomic(ROOT/'gate.json',model)
 atomic(ROOT/'gate_data_index.json',dict(rows=[{k:v for k,v in r.items()if k!='x'}for r in data],gate_sha256=sha(ROOT/'gate.json')))
 print('BENEFIT_STAGE_GATE_TRAINED',json.dumps({k:model[k]for k in ('train_scenes','train_class_counts','initial_loss','final_loss','fail_closed_no_train_rescue')}),flush=True)

def validate():
 collect_plan=json.loads((ROOT/'collect_plan.json').read_text());plan=json.loads((ROOT/'validate_plan.json').read_text());report=json.loads((ROOT/'collection_audit.json').read_text());model=json.loads((ROOT/'gate.json').read_text());cases=[]
 assert model['feature_source_sha256']==sha(D/'benefit_stage_gate.py')and model['training_source_sha256']==sha(__file__)
 for j in plan['jobs']:final_result(Path(j['run']),j['manifest'])
 for case in report['cases']:
  if case['split']!='VAL':continue
  task,seed=case['task'],case['seed'];gjob=next(j for j in plan['jobs']if j['task']==task);g=episode(gjob,seed);proof=audit_episode('G',g)
  chosen='B'if proof['active_queries']else'A0';j=next(j for j in collect_plan['jobs']if j['task']==task and j['branch']==chosen);reference=episode(j,seed);differences=exact(reference,g)
  assert not differences,dict(task=task,seed=seed,chosen=chosen,differences=differences)
  first=case['single_event_query'];expected_scores=None
  if first is not None:
   a0job=next(j for j in collect_plan['jobs']if j['task']==task and j['branch']=='A0');a0=episode(a0job,seed)
   ep,qr,tr,folder,stem=a0
   with np.load(folder/(stem+f'_q{first:03d}.npz'),allow_pickle=False)as z:
    raw=z['raw_delta'];delta=np.zeros((1,32,14),np.float32);delta[0][:,ARM_INDICES]=raw[:,ARM_INDICES]
    packet={k:z['legal__'+k].copy()for k in FORWARD_KEYS|{'base'}};packet['candidate_delta']=delta
   context=next(r['context']for r in qr[first]['multi_event']['records']if r.get('reason')=='new_event_scored')
   from benefit_stage_gate import predict
   expected_scores=predict(model,packet,context)
   actual=next(r['scores']for r in g[1][first]['multi_event']['records']if r.get('reason')=='new_event_scored')
   assert actual==expected_scores,'Deployed gate differs from frozen legal-input model'
   admit=expected_scores['rescue']-2*expected_scores['harm']>0
   assert chosen==('B'if admit else'A0')and proof['active_queries']==([first]if admit else[])
  else:assert not proof['active_queries']
  assert gjob['manifest']['source_sha256'][str(ROOT/'gate.json')]==sha(ROOT/'gate.json')
  cases.append(dict(task=task,seed=seed,selected=chosen,success=g[0]['success'],baseline_success=case['successes']['A0'],always_first_success=case['successes']['B'],exact_selected_branch_replay=True,frozen_model_scores_exact=True,first_event_scores=expected_scores,active_queries=proof['active_queries']))
 assert len(cases)==8
 final=dict(complete=True,passed=True,three_steps_complete=True,collection_episodes=88,gate_validation_episodes=8,total_episodes=96,collection_distribution=report['distribution'],gate_sha256=sha(ROOT/'gate.json'),gate_model=model,validation_cases=cases,validation_successes={k:sum(c[field]for c in cases)for k,field in [('baseline','baseline_success'),('always_first','always_first_success'),('gate','success')]},resources=resource_audit(),independent_EVAL_claim=False,limitations='32 TRAIN and 8 VAL only; no 100-episode efficacy claim, no adapter retraining')
 atomic(ROOT/'final.json',final);atomic(D/'reports/target_benefit_stage_complete_20261026.json',final)
 lines=['# 首事件收益采集、分布与门控验收','',f"完成 40 场景、88 局采集和 8 局门控验证，共 96 局。",'',f"TRAIN/VAL 收益分布：{report['distribution']}",f"VAL 8 场景成功数：{final['validation_successes']}",f"训练无 rescue 时保守拒绝全部干预：{model['fail_closed_no_train_rescue']}。",'','门控验证全轨迹与所选对照分支精确一致；旧 12 局与历史 EVAL 未并入训练。仅为小样本可行性结果。']
 (D/'reports/target_benefit_stage_complete_20261026.md').write_text('\n'.join(lines)+'\n');print('BENEFIT_STAGE_THREE_STEPS_COMPLETE',json.dumps(final['validation_successes']),flush=True)

if __name__=='__main__':
 import sys
 {'collect':collect,'train':train,'validate':validate}[sys.argv[1]]()
