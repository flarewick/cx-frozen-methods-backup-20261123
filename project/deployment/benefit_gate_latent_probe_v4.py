"""CPU-only post-TEST research; TRAIN-cohort CV selects, DEV checks once."""
import argparse,ast,copy,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from benefit_growth_protocol import D,HOME,BASE,sha,atomic
from benefit_growth_learning import packet_at
from benefit_stage_gate import CLASSES
from benefit_gate_latent_v4 import features,DIMS
OUT=HOME/'research_v4'
REGS=(.1,1.,10.,100.)
HARM_WEIGHTS=(1.,2.,4.)

def proof():
 from benefit_gate_latent_v4 import encoder,encode,EXPECTED
 from cx_tta.structured_action_adapter import FORWARD_KEYS
 shapes={'visual':(1,3,2,2,32),'text':(1,2,32),'state':(1,14),'past_visual':(1,3,2,2,32),'past_text':(1,2,32),'past_state':(1,14),'has_history':(1,1),'ee_pose':(1,2,9),'past_ee_pose':(1,2,9),'base':(1,32,14),'candidate_delta':(1,32,14)}
 rng=np.random.default_rng(74139);p={k:rng.normal(0,.1,size=shape).astype(np.float32)for k,shape in shapes.items()}
 p['candidate_delta'][:]=np.clip(p['candidate_delta'],-.1,.1);p['candidate_delta'][:,:,[6,13]]=0;p['has_history'][:]=1
 ctx=np.zeros(8,np.float32);before={k:v.tobytes()for k,v in p.items()};random_before=torch.random.get_rng_state().clone()
 z,a=encode(p);m=encoder();assert torch.equal(random_before,torch.random.get_rng_state())
 state_before={k:v.detach().clone()for k,v in m.state_dict().items()};captured=[]
 hook=m.trunk.register_forward_hook(lambda module,args,output:captured.append(output.detach().clone()))
 try:
  with torch.inference_mode():out=m({k:torch.from_numpy(p[k].copy())for k in FORWARD_KEYS},torch.from_numpy(p['base'].copy()))
 finally:hook.remove()
 assert len(captured)==1 and np.array_equal(z,captured[0][0].numpy().astype(np.float64))
 assert np.array_equal(a,out['current_attention'][0].numpy().reshape(-1).astype(np.float64))
 q={k:v.copy()for k,v in p.items()}
 for k in q:
  if k.startswith('past_')or k=='has_history':q[k][:]=0
 zz,aa=encode(q);assert np.array_equal(z,zz)and np.array_equal(a,aa)
 for family,n in DIMS.items():
  x=features(p,ctx,family);assert x.shape==(n,)and np.isfinite(x).all()and np.array_equal(x,features(p,ctx,family))
 for forbidden in ('task','seed','success','offline_labels','future_state'):
  try:features(dict(p,**{forbidden:None}),ctx,'latent_event72')
  except ValueError:pass
  else:raise AssertionError('Illegal encoder input accepted')
 assert before=={k:v.tobytes()for k,v in p.items()}
 assert all(torch.equal(v,m.state_dict()[k])for k,v in state_before.items())and torch.equal(random_before,torch.random.get_rng_state())
 assert all(not t.requires_grad and t.device.type=='cpu'for t in m.parameters())and not torch.cuda.is_initialized()
 value=dict(passed=True,cpu_only=True,cuda_initialized=False,frozen_original_forward_latent_and_attention_bit_exact=True,current_only_history_mask_exact=True,encoder_parameters_unchanged=True,global_Torch_RNG_unchanged=True,source_packet_unchanged=True,all_privileged_metadata_and_future_inputs_rejected=True,encoder_checkpoint_sha256=EXPECTED,feature_source_sha256=sha(D/'benefit_gate_latent_v4.py'),probe_source_sha256=sha(__file__))
 OUT.mkdir(exist_ok=True);atomic(OUT/'feature_cpu_proof.json',value);return value

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

def fit(rows,family,reg,harm):
 chosen=[r for r in rows if r['features']is not None];x=np.asarray([r['features'][family]for r in chosen],np.float64);labels=np.array([CLASSES.index(r['label'])for r in chosen],np.int64)
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.;z=(x-mean)/scale
 tx=torch.from_numpy(z);ty=torch.from_numpy(labels);weights=torch.ones(len(ty),dtype=torch.float64);weights[ty==2]=harm
 w=torch.zeros((z.shape[1],4),dtype=torch.float64,requires_grad=True);bias=torch.zeros(4,dtype=torch.float64,requires_grad=True)
 optimizer=torch.optim.LBFGS([w,bias],max_iter=120,line_search_fn='strong_wolfe',tolerance_grad=1e-9)
 def closure():
  optimizer.zero_grad();loss=(torch.nn.functional.cross_entropy(tx@w+bias,ty,reduction='none')*weights).sum()/weights.sum()+.5*reg*w.square().sum()/z.shape[1];loss.backward();return loss
 initial=float(closure().detach());optimizer.step(closure);final=float(closure().detach());assert np.isfinite(final)and final<=initial+1e-10
 return dict(schema='frozen_adapter_latent_gate_v4',feature_family=family,regularization=reg,TRAIN_harm_weight=harm,admission_harm_cost=1.,mean=mean.tolist(),scale=scale.tolist(),weight=w.detach().numpy().tolist(),bias=bias.detach().numpy().tolist(),initial_loss=initial,final_loss=final,fit_events=len(chosen),feature_source_sha256=sha(D/'benefit_gate_latent_v4.py'))

def evaluate(model,rows):
 out=[]
 for r in rows:
  if r['features']is None:admit=False
  else:
   x=r['features'][model['feature_family']];z=((x-model['mean'])/model['scale'])@model['weight']+model['bias'];p=np.exp(z-z.max());p/=p.sum();admit=p[1]>p[2]
  out.append((r,admit))
 return dict(scenes=len(rows),baseline=sum(r['a']for r,b in out),ungated=sum(r['b']for r,b in out),gate=sum(r['b']if b else r['a']for r,b in out),rescues=sum(int(b and r['label']=='rescue')for r,b in out),harms=sum(int(b and r['label']=='harm')for r,b in out),admissions=sum(int(b)for r,b in out))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-proof',action='store_true');a=ap.parse_args();torch.set_num_threads(1);value=proof()
 if a.cpu_proof:print('FROZEN_LATENT_CPU_PROOF',json.dumps(value),flush=True);return
 if(OUT/'result.json').exists():raise RuntimeError('Research batch already closed; no repeated DEV search')
 # Enforce TEST isolation at actual Python filesystem reads, not only in metadata.
 forbidden=[(HOME/'test0').resolve(),(HOME/'final.json').resolve(),(HOME/'TEST_outcome_diagnosis.json').resolve()]
 def access(event,args):
  if event!='open'or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=Path(os.fsdecode(args[0])).resolve()
  if any(path==q or q in path.parents for q in forbidden):raise RuntimeError('Forbidden TEST file read: '+str(path))
 sys.addaudithook(access)
 spec=dict(encoder_checkpoint_sha256=__import__('benefit_gate_latent_v4').EXPECTED,feature_families=DIMS,regularizations=REGS,harm_weights=HARM_WEIGHTS,inference_harm_cost=1.,TRAIN_only_fit_and_normalization=True,TRAIN_only_three_cohort_CV_selection=True,DEV_evaluated_once_after_selection=True,TEST_reads_denied=True,selection='CV gain, fewer harms, fewer admissions, lower dimension, larger regularization, lower class weight, enumeration',feature_source_sha256=sha(D/'benefit_gate_latent_v4.py'),probe_source_sha256=sha(__file__))
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
    total={key:sum(f[key]for f in folds)for key in folds[0]};specs.append(dict(family=family,regularization=reg,harm_weight=harm,folds=folds,CV=total))
 best=max(range(len(specs)),key=lambda i:(specs[i]['CV']['gate']-specs[i]['CV']['baseline'],-specs[i]['CV']['harms'],-specs[i]['CV']['admissions'],-DIMS[specs[i]['family']],specs[i]['regularization'],-specs[i]['harm_weight'],-i));s=specs[best];model=fit(sum(groups,[]),s['family'],s['regularization'],s['harm_weight'])
 atomic(OUT/'TRAIN_selected_candidate.json',model);atomic(OUT/'DEV_evaluation_started.json',dict(candidate_sha256=sha(OUT/'TRAIN_selected_candidate.json'),specification_sha256=sha(pp),started_unix=time.time()))
 dev=load(HOME/'dev0','DEV');assert len(dev)==40;result=evaluate(model,dev)
 assert sha(HOME/'gate.json')==original_gate and not torch.cuda.is_initialized()
 report=dict(passed=True,cpu_only=True,TEST_access_denied=True,selection_used_TRAIN_only=True,DEV_checked_once=True,grid=specs,selected=s,DEV=result,ready_for_new_independent_TEST=result['gate']>max(result['baseline'],result['ungated']),original_frozen_gate_unchanged=True,feature_CPU_proof_sha256=sha(OUT/'feature_cpu_proof.json'),preregistration_sha256=sha(pp),candidate_sha256=sha(OUT/'TRAIN_selected_candidate.json'))
 atomic(OUT/'result.json',report);print('FROZEN_LATENT_PROBE_RESULT',json.dumps({k:report[k]for k in('selected','DEV','ready_for_new_independent_TEST','TEST_access_denied')}),flush=True)
if __name__=='__main__':main()
