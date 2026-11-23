"""Real native packet math plus CPU queue-lease fixtures, no actor rollout."""
from pathlib import Path
import dataclasses,hashlib,json
import numpy as np,torch
from native_shared_response_v1.model import LocalSharedResponse
from native_shared_response_v1.session import FeedbackSession,state_sha
from native_delayed_query_v1.predictor import FrozenPredictor
from native_phase_feedback_v1.feedback_queue import Feedback,Ticket
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_shared_response_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def tensor(a):return torch.tensor(np.array(a,copy=True),dtype=torch.float32)
class ReplayQueue:
 """Queue is a software fixture; actual eligibility is inherited from audited GPU packets."""
 def __init__(self,source,queries):self.source=source;self.queries=queries;self.epoch=0
 def at_query(self,*,query,tick,query_proof_sha256):
  record=self.queries[query];assert tick==record['tick']
  return Feedback(Ticket(self.epoch,0),self.source,query,tick,tuple(record['rows']),query_proof_sha256)if record['rows']else None
 def reset(self):self.epoch+=1
 def finish(self):return dict(fixture_only=True)
def run():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
 bfile=T/'outputs/native_command_hidden_v1/binding.json';b=read(bfile);assert b['passed']
 cache=bfile.parent/'query_hidden.npz';assert sha(cache)==b['cache_sha256'];z=load(cache)
 predictor=FrozenPredictor();s=predictor.state
 before=torch.random.get_rng_state().clone()
 model=LocalSharedResponse(z['ARM_weight'],s['context_mean'],s['context_scale'],s['action_scale'])
 initial=state_sha(model.state_dict());hidden=tensor(z['hidden']);arm=[0,1,2,3,4,5,7,8,9,10,11,12]
 logits=tensor(z['logits'][...,arm]);reference=tensor(z['reference'][...,arm])
 qindex={r['file']:i for i,r in enumerate(b['queries'])};bindings={(r['case_id'],r['branch'],r['local_row']):r['hidden_query_index']for r in b['bindings']}
 checks=0;records=[];negative=0
 try:FeedbackSession(model,shadow_only=False)
 except ValueError:checks+=1
 else:raise AssertionError('Untrained response was enabled for actor application')
 recovered=read(T/'outputs/native_online_bridge_recovery_v3/whole_case_audit.json')
 for case in recovered['rows']:
  folder=Path(case['folder'])
  for branch in('Z_reference','C_original','Z_sham'):
   live=read(folder/(branch+'_live_feedback.json'))
   if not live['rows']:continue
   source=live['rows'][0]['command_query'];qq={q['query']:q for q in live['queries']}
   owner=FeedbackSession(model,queue=ReplayQueue(source,qq),shadow_only=True)
   arrays=load(folder/(branch+'_live_local.npz'))
   x,u,k,_=predictor.model.coordinates(arrays['context'],arrays['command'],arrays['start_points'],'correct')
   pp=predictor.predict([dict(context=arrays['context'][i],command=arrays['command'][i],start_points=arrays['start_points'][i])for i in range(len(live['rows']))])
   for q in live['queries']:
    packet=owner.at_query(query=q['query'],tick=q['tick'],query_proof_sha256='a'*64)
    if packet is None:continue
    rr=q['rows'];ids=[bindings[(case['case_id'],branch,i)]for i in rr]
    suffix='following'if q['query']==source+1 else'delayed';qi=qindex[str(folder/(branch+'_'+suffix+'_query.npz'))]
    values=dict(row_ids=rr,hidden=hidden[ids],context=tensor(x[rr]),command=tensor(u[rr]),kind=tensor(k[rr]),
      kernel_prediction=tensor(pp[rr]),target=tensor(arrays['target'][rr]),next_hidden=hidden[qi:qi+1],
      next_logits=logits[qi:qi+1],next_reference=reference[qi:qi+1])
    copied=dataclasses.replace(packet)
    try:owner.consume(copied,**values)
    except ValueError:checks+=1
    else:raise AssertionError('Copied/cross-session packet accepted')
    output,report=owner.consume(packet,**values)
    assert torch.equal(output,reference[qi:qi+1])and not report['applied_to_actor'];checks+=2
    try:owner.consume(packet,**values)
    except ValueError:checks+=1
    else:raise AssertionError('Packet consumed twice')
    # Real data with one explicit nonfinite response tests rejection/rollback.
    bad=FeedbackSession(model,queue=ReplayQueue(source,qq),shadow_only=True)
    lease=bad.at_query(query=q['query'],tick=q['tick'],query_proof_sha256='b'*64)
    corrupted=dict(values,target=values['target'].clone());corrupted['target'][0,0]=float('nan')
    returned,rejected=bad.consume(lease,**corrupted)
    assert not rejected['accepted']and rejected['fast_before']==rejected['fast_after']and torch.equal(returned,reference[qi:qi+1]);checks+=3;negative+=1
    bad.finish();assert all(not bool(v.any())for v in bad.fast)and bad.closed;checks+=2
    owner.reset();assert all(not bool(v.any())for v in owner.fast)and not owner.anchors and owner.attempts==0;checks+=3
    try:owner.consume(packet,**values)
    except ValueError:checks+=1
    else:raise AssertionError('Previous-episode packet survived reset')
    budget=FeedbackSession(model,queue=ReplayQueue(source,qq),shadow_only=True)
    lease=budget.at_query(query=q['query'],tick=q['tick'],query_proof_sha256='c'*64);budget.attempts=3
    try:budget.consume(lease,**values)
    except ValueError:checks+=1
    else:raise AssertionError('Fourth source update accepted')
    records.append(dict(case_id=case['case_id'],branch=branch,**report))
 assert len(records)==12 and any(r['accepted']for r in records);checks+=1
 assert state_sha(model.state_dict())==initial and all(p.requires_grad for p in model.response.parameters());checks+=2
 assert torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized();checks+=2
 report=dict(passed=True,checks=checks,actual_audited_packets=len(records),records=records,
   nonfinite_rollback_cases=negative,CPU_queue_fixture=True,actual_SAM2_simulator_or_task_rollout=False,
   original_model_RNG_unchanged=True,untrained_actor_application_rejected=True,shadow_outputs_frozen_exact=True,
   source_sha256={str(p):sha(p)for p in(C/'model.py',C/'session.py',Path(__file__))},
   input_sha256={str(bfile):sha(bfile),str(cache):sha(cache),str(O/'interface.json'):sha(O/'interface.json')},
   new_independent_scenes=0,model_fit_steps=0,online_updates=0,TTA_utility_proven=False)
 with(O/'session_checks.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print('ACTUAL_PACKET_SHARED128_STATE_SHADOW_ROLLBACK_PASS',checks,len(records),sum(r['accepted']for r in records),flush=True)
if __name__=='__main__':run()
