"""CPU-only post-TEST research; TRAIN-cohort CV selects, DEV checks once."""
import argparse,ast,copy,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from benefit_growth_protocol import D,HOME,BASE,sha,atomic
from benefit_growth_learning import packet_at
from benefit_stage_gate import CLASSES
from benefit_gate_paired_v5 import features,DIMS
OUT=HOME/'research_v5'
REGS=(.1,1.,10.)
HARM_WEIGHTS=(1.,4.,16.)

def proof():
 import copy
 from benefit_gate_paired_v5 import contrast
 rows=[]
 for value,a,b in [(-1.,1,0),(-.5,1,0),(.5,0,1),(1.,0,1)]:
  x=np.zeros(50,np.float64);x[0]=value;rows.append(dict(a=a,b=b,label='harm'if a else'rescue',features={'compact50':x}))
 m=fit(rows,'compact50',1.,4.);e=evaluate(m,rows);assert e['gate']==4 and e['harms']==0 and e['rescues']==2
 swapped=[dict(r,a=r['b'],b=r['a'],label='rescue'if r['a']else'harm')for r in rows];n=fit(swapped,'compact50',1.,4.)
 assert np.allclose(np.asarray(m['effect_weight']),-np.asarray(n['effect_weight']),atol=1e-8,rtol=0)and abs(m['effect_bias']+n['effect_bias'])<1e-8
 neutral=[dict(r,b=r['a'],label='both_success'if r['a']else'both_fail')for r in rows];z=fit(neutral,'compact50',1.,4.)
 assert np.max(np.abs(z['effect_weight']))<1e-10 and abs(z['effect_bias'])<1e-10
 assert not torch.cuda.is_initialized()
 OUT.mkdir(exist_ok=True);value=dict(passed=True,cpu_only=True,cuda_initialized=False,paired_outcome_direction_recovered=True,swapped_treatment_outcomes_reverse_effect=True,neutral_pairs_have_zero_effect=True,centered_shared_success_model=True,feature_source_sha256=sha(D/'benefit_gate_paired_v5.py'),probe_source_sha256=sha(__file__))
 atomic(OUT/'feature_cpu_proof.json',value);return value

def load(root,split):
 a=json.loads((root/'collection_audit.json').read_text());assert a['passed']and a['complete'];plan=json.loads((root/'collect_plan.json').read_text());rows=[]
 for c in a['cases']:
  if c['split']!=split:continue
  packet=packet_at(plan,c);label=c['benefit_label']
  if packet is None:
   assert c['successes']['A0']==c['successes']['B'];label='both_success'if c['successes']['A0']else'both_fail';f=None
  else:f={family:features(*packet,family)for family in DIMS}
  rows.append(dict(a=int(c['successes']['A0']),b=int(c['successes']['B']),label=label,features=f))
 return rows

def fit(rows,family,reg,effect_ratio):
 chosen=[r for r in rows if r['features']is not None];x=np.asarray([r['features'][family]for r in chosen],np.float64)
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.;z=(x-mean)/scale;tx=torch.from_numpy(z);ty=torch.tensor([[r['a'],r['b']]for r in chosen],dtype=torch.float64)
 w=torch.zeros(z.shape[1],dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True);u=torch.zeros_like(w,requires_grad=True);ub=torch.zeros((),dtype=torch.float64,requires_grad=True)
 opt=torch.optim.LBFGS([w,bias,u,ub],max_iter=120,line_search_fn='strong_wolfe',tolerance_grad=1e-9)
 def closure():
  opt.zero_grad();eta=tx@w+bias;tau=tx@u+ub;logits=torch.stack((eta-tau/2,eta+tau/2),dim=1)
  loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,ty)+.5*reg*(w.square().sum()+effect_ratio*u.square().sum())/z.shape[1];loss.backward();return loss
 initial=float(closure().detach());opt.step(closure);final=float(closure().detach());assert np.isfinite(final)and final<=initial+1e-10
 return dict(schema='paired_success_candidate_gate_v5',feature_family=family,regularization=reg,effect_penalty_ratio=effect_ratio,mean=mean.tolist(),scale=scale.tolist(),baseline_weight=w.detach().numpy().tolist(),baseline_bias=float(bias.detach()),effect_weight=u.detach().numpy().tolist(),effect_bias=float(ub.detach()),initial_loss=initial,final_loss=final,fit_pairs=len(chosen),feature_source_sha256=sha(D/'benefit_gate_paired_v5.py'))

def evaluate(model,rows):
 from benefit_gate_paired_v5 import contrast
 out=[]
 for r in rows:
  admit=False if r['features']is None else contrast(model,r['features'][model['feature_family']])>0
  out.append((r,admit))
 return dict(scenes=len(rows),baseline=sum(r['a']for r,b in out),ungated=sum(r['b']for r,b in out),gate=sum(r['b']if b else r['a']for r,b in out),rescues=sum(int(b and r['label']=='rescue')for r,b in out),harms=sum(int(b and r['label']=='harm')for r,b in out),admissions=sum(int(b)for r,b in out))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-proof',action='store_true');a=ap.parse_args();torch.set_num_threads(1);value=proof()
 if a.cpu_proof:print('PAIRED_SUCCESS_CPU_PROOF',json.dumps(value),flush=True);return
 if(OUT/'result.json').exists():raise RuntimeError('Research batch already closed; no repeated DEV search')
 # Enforce TEST isolation at actual Python filesystem reads, not only in metadata.
 forbidden=[(HOME/'test0').resolve(),(HOME/'final.json').resolve(),(HOME/'TEST_outcome_diagnosis.json').resolve()]
 def access(event,args):
  if event!='open'or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).resolve()
  if any(path==q or q in path.parents for q in forbidden):raise RuntimeError('Forbidden TEST file read: '+str(path))
 sys.addaudithook(access)
 spec=dict(feature_families=DIMS,regularizations=REGS,effect_penalty_ratios=HARM_WEIGHTS,inference_expected_success_contrast_threshold=0.,TRAIN_only_fit_and_normalization=True,TRAIN_only_three_cohort_CV_selection=True,DEV_evaluated_once_after_selection=True,TEST_reads_denied=True,selection='CV gain, fewer harms, fewer admissions, lower dimension, larger regularization, lower effect penalty ratio, enumeration',feature_source_sha256=sha(D/'benefit_gate_paired_v5.py'),probe_source_sha256=sha(__file__))
 pp=OUT/'preregistration.json'
 if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(spec)),'Research preregistration changed'
 else:atomic(pp,spec)
 original_gate=sha(HOME/'gate.json');groups=[load(BASE,'TRAIN'),load(HOME/'train0','TRAIN'),load(HOME/'train1','TRAIN')];assert list(map(len,groups))==[32,40,40]
 specs=[]
 for family in DIMS:
  for reg in REGS:
   for harm in HARM_WEIGHTS:
    folds=[]
    for k in range(3):folds.append(evaluate(fit(sum([g for i,g in enumerate(groups)if i!=k],[]),family,reg,harm),groups[k]))
    total={key:sum(f[key]for f in folds)for key in folds[0]};specs.append(dict(family=family,regularization=reg,effect_penalty_ratio=harm,folds=folds,CV=total))
 best=max(range(len(specs)),key=lambda i:(specs[i]['CV']['gate']-specs[i]['CV']['baseline'],-specs[i]['CV']['harms'],-specs[i]['CV']['admissions'],-DIMS[specs[i]['family']],specs[i]['regularization'],-specs[i]['effect_penalty_ratio'],-i));s=specs[best];model=fit(sum(groups,[]),s['family'],s['regularization'],s['effect_penalty_ratio'])
 atomic(OUT/'TRAIN_selected_candidate.json',model);atomic(OUT/'DEV_evaluation_started.json',dict(candidate_sha256=sha(OUT/'TRAIN_selected_candidate.json'),specification_sha256=sha(pp),started_unix=time.time()))
 dev=load(HOME/'dev0','DEV');assert len(dev)==40;result=evaluate(model,dev)
 assert sha(HOME/'gate.json')==original_gate and not torch.cuda.is_initialized()
 report=dict(passed=True,cpu_only=True,TEST_access_denied=True,selection_used_TRAIN_only=True,DEV_checked_once=True,grid=specs,selected=s,DEV=result,ready_for_new_independent_TEST=result['gate']>max(result['baseline'],result['ungated']),original_frozen_gate_unchanged=True,feature_CPU_proof_sha256=sha(OUT/'feature_cpu_proof.json'),preregistration_sha256=sha(pp),candidate_sha256=sha(OUT/'TRAIN_selected_candidate.json'))
 atomic(OUT/'result.json',report);print('PAIRED_SUCCESS_PROBE_RESULT',json.dumps({k:report[k]for k in('selected','DEV','ready_for_new_independent_TEST','TEST_access_denied')}),flush=True)
if __name__=='__main__':main()
