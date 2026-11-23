"""Real12 offline packets validate online-only feature binding; no teacher files opened."""
from pathlib import Path
import json,copy,hashlib
import numpy as np,torch
from .bind import inputs,ARM,tensor
from native_delayed_query_v1.predictor import FrozenPredictor
from native_phase_feedback_v1.feedback_queue import Feedback,Ticket
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def main():
 torch.set_num_threads(2);predictor=FrozenPredictor();before=torch.random.get_rng_state().clone()
 index=read(T/'outputs/native_expert_full_feedback_v1/index.json');cases={c['case_id']:c for c in index['cases']}
 packets=read(T/'outputs/native_expert_packets_v1/manifest.json')['packets'];checked=[];negative=0
 for number,p in enumerate(packets):
  local=load(p['local_file']);queries=cases[p['case_id']]['queries'];byq={q['query']:q for q in queries};current_data=load(p['next_policy_file'])
  packet=Feedback(Ticket(0,number),p['source_query'],p['query'],p['tick'],tuple(p['row_ids']),'a'*64)
  rows={i:dict(context=local['context'][i],command=local['command'][i],start_points=local['start_points'][i],target=local['target'][i],command_query=int(local['command_query'][i]),start_tick=int(local['start_tick'][i]),end_tick=int(local['end_tick'][i]))for i in packet.rows}
  issuing={q:dict(query=q,tick=byq[q]['tick'],hidden=load(byq[q]['policy']['policy_file'])['hidden'])for q in {r['command_query']for r in rows.values()}}
  current=dict(query=p['query'],tick=p['tick'],query_proof_sha256='a'*64,**{k:current_data[k]for k in('hidden','logits','raw_delta')})
  actual=inputs(packet,rows=rows,issuing_queries=issuing,current=current,predictor=predictor)
  expected=dict(hidden=local['issuing_query_hidden'][p['row_ids']],context=local['model_context'][p['row_ids']],command=local['model_command'][p['row_ids']],kind=local['model_kind'][p['row_ids']],kernel_prediction=local['kernel_prediction'][p['row_ids']],target=local['target'][p['row_ids']],next_hidden=current_data['hidden'][None],next_logits=current_data['logits'][:,ARM][None],next_reference=current_data['raw_delta'][:,ARM][None])
  for key,value in expected.items():assert torch.equal(actual[key],tensor(value)),key
  poison=dict(current,teacher_target=object(),task_id=object(),seed=object(),future_frames=object())
  repeated=inputs(packet,rows=rows,issuing_queries=issuing,current=poison,predictor=predictor)
  for key in expected:assert torch.equal(actual[key],repeated[key])
  for mode in ('future_window','future_command','wrong_hidden_owner','wrong_query_proof'):
   rr=copy.deepcopy(rows);ii=copy.deepcopy(issuing);cc=dict(current);first=packet.rows[0]
   if mode=='future_window':rr[first]['end_tick']=packet.observation_tick+1
   elif mode=='future_command':rr[first]['command_query']=packet.query
   elif mode=='wrong_hidden_owner':ii[rr[first]['command_query']]['query']+=1
   else:cc['query_proof_sha256']='b'*64
   try:inputs(packet,rows=rr,issuing_queries=ii,current=cc,predictor=predictor)
   except ValueError:negative+=1
   else:raise AssertionError(mode+' accepted')
  checked.append(dict(case_id=p['case_id'],source_query=p['source_query'],query=p['query'],local_rows=len(packet.rows),actual_issuing_hidden_and_model_inputs_exact=True))
 assert len(checked)==12 and negative==48 and torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 out=T/'outputs/native_response_binding_v1';out.mkdir(exist_ok=True)
 report=dict(passed=True,real_saved_packets=checked,negative_bindings_rejected=negative,teacher_task_seed_future_poison_has_no_effect=True,teacher_target_files_read=0,actor_updates=0,GPU_forwards=0,TTA_utility_proven=False,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 with(out/'CPU_live_input_binding_checks.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print('ACTUAL_ISSUING_QUERY_ONLINE_INPUT_BINDING_PASS',12,negative,flush=True)
if __name__=='__main__':main()
