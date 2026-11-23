"""Actual saved data exercise derivative plumbing; no fitting or deployable checkpoint."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from model import LocalSharedResponse
from native_delayed_query_v1.predictor import FrozenPredictor
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_shared_response_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def write(p,value):
 with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def tensor(a):return torch.as_tensor(np.array(a,copy=True),dtype=torch.float32)
def run():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
 bindingfile=T/'outputs/native_command_hidden_v1/binding.json';binding=read(bindingfile);assert binding['passed']
 file=bindingfile.parent/'query_hidden.npz';assert sha(file)==binding['cache_sha256'];saved=load(file)
 recoveryfile=T/'outputs/native_online_bridge_recovery_v3/whole_case_audit.json';recovery=read(recoveryfile);assert recovery['passed']
 inputs={str(bindingfile):sha(bindingfile),str(file):sha(file),str(recoveryfile):sha(recoveryfile)}
 for p,h in binding['input_sha256'].items():assert sha(p)==h
 predictor=FrozenPredictor();state=predictor.state
 O.mkdir(parents=True,exist_ok=True)
 protocol=dict(interface_only=True,actual_rows=binding['rows'],actual_queries=len(binding['queries']),
   input_sha256=inputs,source_sha256={str(p):sha(p)for p in(C/'model.py',Path(__file__))},
   model='122-64-388 bilinear local command194; zero-matched frozen kernel',
   target_scale=.003125,fast_parameters=128,inner_steps=2,lr=.05,clip=1.,seed=17,
   no_hyperparameter_search=True,model_fit_steps=0,new_independent_scenes=0,online_updates=0,TTA_utility_proven=False)
 write(O/'protocol.json',protocol)
 before=torch.random.get_rng_state().clone()
 model=LocalSharedResponse(saved['ARM_weight'],state['context_mean'],state['context_scale'],state['action_scale'])
 assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 model.eval();original={k:v.clone()for k,v in model.state_dict().items()};checks=0;total=0;packets=[];unknown=[]
 refs=tensor(saved['reference'][...,[0,1,2,3,4,5,7,8,9,10,11,12]])
 logits=tensor(saved['logits'][...,[0,1,2,3,4,5,7,8,9,10,11,12]])
 hidden=tensor(saved['hidden']);fast=model.zero_fast()
 assert torch.equal(model.direction(hidden,logits,refs,fast),refs);checks+=len(refs)
 lookup={(r['case_id'],r['branch'],r['local_row']):r for r in binding['bindings']}
 queryfiles={q['file']:i for i,q in enumerate(binding['queries'])}
 for record in recovery['rows']:
  folder=Path(record['folder'])
  for branch in('Z_reference','C_original','Z_sham'):
   livefile=folder/(branch+'_live_feedback.json');live=read(livefile);inputs[str(livefile)]=sha(livefile)
   if not live['rows']:
    unknown.append(dict(case_id=record['case_id'],branch=branch,reason='no_local_rows_retained'));continue
   a=load(folder/(branch+'_live_local.npz'))
   inputs[str(folder/(branch+'_live_local.npz'))]=sha(folder/(branch+'_live_local.npz'))
   ids=np.array([lookup[(record['case_id'],branch,i)]['hidden_query_index']for i in range(len(live['rows']))])
   rows=[dict(context=a['context'][i],command=a['command'][i],start_points=a['start_points'][i])for i in range(len(ids))]
   xx,v,kind,_=predictor.model.coordinates(a['context'],a['command'],a['start_points'],'correct')
   pred=predictor.predict(rows);x,u,k,p,y=map(tensor,(xx,v,kind,pred,a['target']))
   result=model(hidden[ids],x,u,k,p,model.zero_fast())
   assert torch.equal(result,p);checks+=len(ids);total+=len(ids)
   null=model(hidden[ids],x,torch.zeros_like(u),k,torch.zeros_like(p),(torch.full((64,),.01),torch.full((64,),-.01)))
   assert torch.equal(null,torch.zeros_like(null));checks+=len(ids)
   used=set()
   for boundary in live['queries']:
    rr=boundary['rows']
    if not rr:continue
    assert not used.intersection(rr);used.update(rr);checks+=1
    for i in rr:assert live['rows'][i]['end_tick']<=boundary['tick'];checks+=1
    suffix='following'if boundary['query']==live['rows'][0]['command_query']+1 else'delayed'
    nextfile=str(folder/(branch+'_'+suffix+'_query.npz'))
    assert nextfile in queryfiles;qi=queryfiles[nextfile]
    assert binding['queries'][qi]['query']==boundary['query'];checks+=1
    f,info=model.two_steps(hidden[ids[rr]],x[rr],u[rr],k[rr],p[rr],y[rr],create_graph=True)
    actual_next=model.direction(hidden[qi:qi+1],logits[qi:qi+1],refs[qi:qi+1],f)
    # Pure derivative probe, not an invented expert direction or training loss.
    scalar=(actual_next-refs[qi:qi+1]).sum()
    grads=torch.autograd.grad(scalar,tuple(model.response.parameters()),allow_unused=False)
    gn=float(torch.sqrt(sum(g.square().sum()for g in grads)))
    assert np.isfinite(gn)and gn>0;checks+=1
    assert all(np.isfinite(info[z]).all()for z in('losses','gradient_norms'));checks+=1
    accepted,trust=model.trust(hidden[qi:qi+1],logits[qi:qi+1],refs[qi:qi+1],f)
    packets.append(dict(case_id=record['case_id'],branch=branch,query=boundary['query'],rows=len(rr),
      actual_command_queries=sorted(set(live['rows'][i]['command_query']for i in rr)),
      derivative_probe_only=True,outer_to_response_gradient_norm=gn,inner=info,trust=trust,
      random_interface_update_would_pass_trust=accepted,applied_to_actor=False))
 accepted,_=model.trust(hidden,logits,refs,model.zero_fast());assert accepted;checks+=1
 accepted,_=model.trust(hidden,logits,refs,(torch.ones(64),torch.ones(64)));assert not accepted;checks+=1
 try:model.two_steps(hidden[:1],torch.zeros(1,53),torch.zeros(1,194),torch.zeros(1,5),torch.zeros(1,2),torch.zeros(1,2),online=True)
 except ValueError:checks+=1
 else:raise AssertionError('Unfrozen random response online was not rejected')
 assert total==binding['rows']and all(torch.equal(v,original[k])for k,v in model.state_dict().items());checks+=1
 assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized();checks+=1
 for group in('source_sha256','input_sha256'):
  for p,h in protocol[group].items():assert sha(p)==h
 # Add all data actually used, independently of the predeclared core input manifest.
 write(O/'interface.json',dict(passed=True,checks=checks,actual_rows=total,queries=len(refs),packets=packets,unknown=unknown,
   all_actual_command_query_bound=True,zero_fast_prediction_and_direction_exact=True,physical_null_exact=True,
   models_and_RNG_unchanged=True,CPU_only=True,CUDA_initialized=False,
   source_sha256=protocol['source_sha256'],input_sha256=inputs,protocol_sha256=sha(O/'protocol.json'),
   model_fit_steps=0,optimizer_steps=0,software_inner_steps_not_training=2*len(packets),online_updates=0,
   new_independent_scenes=0,full_candidate_frozen=False,direction_training_allowed=False,TTA_utility_proven=False))
 print('ACTUAL_NATIVE_LOCAL_SHARED128_INTERFACE_PASS',checks,total,len(packets),flush=True)
if __name__=='__main__':run()
