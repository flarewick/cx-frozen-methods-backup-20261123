"""CPU-only fitting; new DEV selects; new TEST cannot influence the fit."""
import json,os,gzip
from pathlib import Path
import numpy as np
from benefit_growth_protocol import D,BASE,HOME,sha,atomic
from benefit_stage_gate import CLASSES
from benefit_pilot_audit import episode
from cx_tta.structured_action_adapter import ARM_INDICES,FORWARD_KEYS
from benefit_growth_gate import features

def packet_at(plan,case):
 job=next(j for j in plan['jobs']if j['task']==case['task']and j['branch']=='A0')
 first=case['single_event_query']
 if first is None:return None
 run=Path(job['run']);journal=json.loads((run/'journal.json').read_text());ep=next(e for e in journal['episodes']if e['seed']==case['seed'])
 session=run/'sessions'/ep['process_session'];stem=f"episode{ep['episode_index']:03d}_seed{case['seed']}";folder=session/'pilot_traces'
 event_record=None
 with gzip.open(session/'queries'/(stem+'.jsonl.gz'),'rt')as stream:
  for line in stream:
   row=json.loads(line)
   if row['query']==first:event_record=row;break
 if event_record is None:raise RuntimeError('Missing frozen first-event query')
 with np.load(folder/(stem+f'_q{first:03d}.npz'),allow_pickle=False)as z:
  delta=np.zeros((1,32,14),np.float32);delta[0][:,ARM_INDICES]=z['raw_delta'][:,ARM_INDICES]
  packet={k:z['legal__'+k].copy()for k in FORWARD_KEYS|{'base'}};packet['candidate_delta']=delta
 context=next(r['context']for r in event_record['multi_event']['records']if r.get('reason')=='new_event_scored')
 return packet,context

def data(root,allowed=None):
 report=json.loads((root/'collection_audit.json').read_text());assert report['passed']
 plan=json.loads((root/'collect_plan.json').read_text());out=[]
 for c in report['cases']:
  if allowed is not None and c['split']not in allowed:continue
  value=packet_at(plan,c)
  if value is None:
   assert c['successes']['A0']==c['successes']['B']
   label='both_success'if c['successes']['A0']else'both_fail'
   feature=None
  else:
   assert c['benefit_label']is not None
   packet,context=value;label=c['benefit_label']
   feature={k:features(packet,context,k)for k in ('compact','channels')}
  out.append(dict(task=c['task'],seed=c['seed'],split=c['split'],label=label,a=int(c['successes']['A0']),b=int(c['successes']['B']),features=feature))
 return out

def fit(rows,family,method,regularization,gamma=1.):
 rows=[r for r in rows if r['features']is not None]
 x=np.asarray([r['features'][family]for r in rows],np.float64);y=np.asarray([CLASSES.index(r['label'])for r in rows],np.int64)
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1;xx=(x-mean)/scale
 model=dict(schema='candidate_utility_gate_v2',feature_family=family,method=method,regularization=regularization,mean=mean.tolist(),scale=scale.tolist(),harm_cost=1.,classes=list(CLASSES),class_prior=(np.bincount(y,minlength=4)/len(y)).tolist())
 if method=='softmax':
  import torch
  torch.set_num_threads(1);assert not torch.cuda.is_initialized()
  tx=torch.from_numpy(xx);ty=torch.from_numpy(y)
  w=torch.zeros((xx.shape[1],4),dtype=torch.float64,requires_grad=True);bias=torch.zeros(4,dtype=torch.float64,requires_grad=True)
  opt=torch.optim.LBFGS([w,bias],max_iter=100,line_search_fn='strong_wolfe',tolerance_grad=1e-9)
  def closure():
   opt.zero_grad();loss=torch.nn.functional.cross_entropy(tx@w+bias,ty)+.5*regularization*w.square().sum()/xx.shape[1]
   loss.backward();return loss
  initial=float(closure().detach());opt.step(closure);final=float(closure().detach());assert np.isfinite(final)and final<=initial+1e-10
  model.update(weight=w.detach().numpy().tolist(),bias=bias.detach().numpy().tolist(),initial_loss=initial,final_loss=final)
 elif method=='kernel_ridge':
  distance=np.maximum((xx**2).sum(1)[:,None]+(xx**2).sum(1)[None,:]-2*xx@xx.T,0)/xx.shape[1]
  kernel=np.exp(-gamma*distance);coef=np.linalg.solve(kernel+regularization*np.eye(len(x)),np.eye(4)[y])
  model.update(train_x=xx.tolist(),coefficient=coef.tolist(),gamma=gamma)
 else:raise ValueError(method)
 return model

def _event_probabilities(model,rows):
 x=np.asarray([r['features'][model['feature_family']]for r in rows]);xx=(x-model['mean'])/model['scale']
 if model['method']=='softmax':
  z=xx@model['weight']+model['bias'];p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)
 train=np.asarray(model['train_x']);dist=np.maximum((xx**2).sum(1)[:,None]+(train**2).sum(1)[None,:]-2*xx@train.T,0)/xx.shape[1]
 p=np.clip(np.exp(-model['gamma']*dist)@model['coefficient'],0,1);small=p.sum(1)<1e-12;p[small]=model['class_prior'];return p/p.sum(1,keepdims=True)

def batch_probabilities(model,rows):
 out=np.zeros((len(rows),4),np.float64);out[:,0]=1
 indices=[i for i,r in enumerate(rows)if r['features']is not None]
 if indices:out[indices]=_event_probabilities(model,[rows[i]for i in indices])
 return out

def main():
 if (HOME/'gate.json').exists():raise RuntimeError('Gate already frozen; do not refit')
 # Neither old VAL nor any TEST scenes are loaded by fitting or selection.
 train=data(BASE,{'TRAIN'})+data(HOME/'train0',{'TRAIN'})+data(HOME/'train1',{'TRAIN'})
 dev=data(HOME/'dev0',{'DEV'});assert len(train)==112 and len(dev)==40
 candidates=[];models=[]
 for family in ('compact','channels'):
  specs=[('softmax',reg,1.)for reg in (.01,.1,1.,10.)]+[('kernel_ridge',reg,gamma)for reg in (.01,.1,1.)for gamma in (.1,1.,10.)]
  for method,reg,gamma in specs:
   model=fit(train,family,method,reg,gamma);prob=batch_probabilities(model,dev)
   for cost in (.5,1.,2.):
    admit=prob[:,1]-cost*prob[:,2]>0;success=sum(r['b']if a else r['a']for r,a in zip(dev,admit))
    rescued=sum(r['label']=='rescue'and a for r,a in zip(dev,admit));harmed=sum(r['label']=='harm'and a for r,a in zip(dev,admit))
    spec=dict(feature_family=family,method=method,regularization=reg,gamma=gamma,harm_cost=cost,successes=int(success),admissions=int(admit.sum()),rescues=int(rescued),harms=int(harmed))
    candidates.append(spec);models.append(dict(model,harm_cost=cost))
 # Fixed deterministic tie break: success, fewer harms, fewer admissions,
 # then enumeration order. No TEST access or task/seed lookup.
 best=max(range(len(candidates)),key=lambda i:(candidates[i]['successes'],-candidates[i]['harms'],-candidates[i]['admissions'],-i))
 selected=candidates[best];model=models[best]
 baseline=sum(r['a']for r in dev);ungated=sum(r['b']for r in dev)
 ready=selected['successes']>max(baseline,ungated)and selected['rescues']>0
 audit=dict(passed=True,train_scenes=len(train),train_counts={k:sum(r['label']==k for r in train)for k in CLASSES},DEV_scenes=len(dev),DEV_baseline=baseline,DEV_ungated=ungated,candidates=candidates,selected=selected,ready_for_independent_TEST=ready,TEST_loaded=False,old_VAL_loaded=False,training_source_sha256=sha(__file__),feature_source_sha256=sha(D/'benefit_growth_gate.py'),data_audit_sha256={str(r):sha(r/'collection_audit.json')for r in (BASE,HOME/'train0',HOME/'train1',HOME/'dev0')})
 atomic(HOME/'selection.json',audit)
 if ready:
  model.update(train_scenes=len(train),train_class_counts=audit['train_counts'],feature_source_sha256=audit['feature_source_sha256'],training_source_sha256=sha(__file__),selection_sha256=sha(HOME/'selection.json'),validation_used_for_fit=False,DEV_used_for_selection=True,TEST_loaded=False,score_semantics='normalized risk-weighted class scores; not calibrated probabilities')
  atomic(HOME/'gate.json',model)
 print('GROWTH_GATE_SELECTION',json.dumps({k:audit[k]for k in ('train_scenes','train_counts','DEV_baseline','DEV_ungated','selected','ready_for_independent_TEST')}),flush=True)
if __name__=='__main__':main()
