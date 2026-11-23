"""Qualified weights on saved real data; software controls, no closed-loop claim."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from .bridge import load_model,DirectionBridge
from native_response_action_v1.compose import EventComposer,ARM
from native_shared_response_v1.session import state_sha
from native_phase_feedback_v1.feedback_queue import Feedback,Ticket
from native_delayed_query_v1.predictor import FrozenPredictor
from cx_tta.multi_event import EventConfig
T=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
class ReplayQueue:
 def __init__(self,packets,queries):self.packets={p['query']:p for p in packets};self.queries=queries;self.epoch=0;self.last=-1;self.released=0
 def at_query(self,*,query,tick,query_proof_sha256):
  assert query==self.last+1 and tick==self.queries[query]['tick'];self.last=query
  p=self.packets.get(query)
  if p is None:return None
  self.released+=1
  return Feedback(Ticket(self.epoch,self.released-1),p['source_query'],query,tick,tuple(p['row_ids']),query_proof_sha256)
 def finish(self):return dict(software_replay_fixture=True,released_feedback_packets=self.released)
 def reset(self):self.epoch+=1;self.last=-1;self.released=0

def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');before=torch.random.get_rng_state().clone()
 files=list((T/'outputs/native_direction_visible_training_v1/runs').glob('*/qualification.json'));assert len(files)==1;qualification=files[0]
 model,q=load_model(qualification);initial=state_sha(model.state_dict());predictor=FrozenPredictor()
 index=read(T/'outputs/native_expert_full_feedback_v1/index.json');old=read(T/'outputs/native_expert_packets_v1/manifest.json');config=EventConfig(**read(T/'outputs/native_expert_full_cache_v8/protocol.json')['event_config'])
 records=[];queries=changed=accepted=0;consumed=0
 for case in [c for c in index['cases']if c['old_pilot_reused']]:
  refs={r['query']:r for r in case['queries']};packets=[p for p in old['packets']if p['case_id']==case['case_id']]
  modes={mode:DirectionBridge(model,qualification,mode=mode,queue=ReplayQueue(packets,refs))for mode in ('shadow','TTA')}
  composers={mode:EventComposer(config)for mode in ('full','zero','shadow','TTA')};rows={};issuing={};updates=[]
  for p in packets:
   z=load(p['local_file']);rows[p['source_query']]={i:dict(context=z['context'][i],command=z['command'][i],target=z['target'][i],start_points=z['start_points'][i],start_tick=int(z['start_tick'][i]),end_tick=int(z['end_tick'][i]),command_query=int(z['command_query'][i]))for i in p['row_ids']}
  for query,ref in refs.items():
   data=load(ref['policy']['policy_file']);current=dict(query=query,tick=ref['tick'],query_proof_sha256='a'*64,**{k:data[k]for k in ('hidden','logits','raw_delta')})
   legal={k[len('legal_'):]:v for k,v in data.items()if k.startswith('legal_')};base=data['base'];raw=data['raw_delta']
   a,_,_=composers['full'].step(base,raw,legal)
   with torch.no_grad():zero=model.direction(torch.tensor(data['hidden'][None]),torch.tensor(data['logits'][:,ARM][None]),torch.tensor(raw[:,ARM][None]),model.zero_fast()).numpy()[0]
   z,_,_=composers['zero'].step(base,raw,legal,adapted_ARM=zero);assert a.tobytes()==z.tobytes()
   outputs={}
   for mode,bridge in modes.items():
    direction,report=bridge.step(current=current,rows_by_source=rows,issuing_queries=issuing,predictor=predictor)
    out,_,metadata=composers[mode].step(base,raw,legal,adapted_ARM=direction);outputs[mode]=out
    if mode=='TTA':
     updates.append(report);accepted+=bool(report['accepted']);consumed+=report['reason']!='no_new_packet'
     changed+=not np.array_equal(out,a)
    else:assert out.tobytes()==a.tobytes()
   assert all(np.array_equal(o[:,[6,13]],base[:,[6,13]])for o in outputs.values())
   issuing[query]=dict(query=query,tick=ref['tick'],hidden=data['hidden'].copy());queries+=1
  for bridge in modes.values():
   finish=bridge.finish();assert finish['fast_reset_on_close']and finish['action_visible_fixed_parameters_and_buffers_unchanged']and bridge.session.closed and all(not bool(v.any())for v in bridge.session.fast)
  records.append(dict(case_id=case['case_id'],queries=len(refs),updates=updates))
 assert queries==36 and consumed==12 and accepted>0 and changed>0
 assert state_sha(model.state_dict())==initial and torch.equal(before,torch.random.get_rng_state())and not torch.cuda.is_initialized()
 out=T/'outputs/native_visible_response_bridge_v1';out.mkdir(exist_ok=True)
 report=dict(passed=True,fixed_step1000_qualification=str(qualification),qualification_sha256=sha(qualification),actual_saved_queries=queries,saved_feedback_packets_consumed=consumed,accepted_software_updates=accepted,changed_saved_query_action_arrays=changed,zero_and_shadow_actions_byte_exact=True,gripper_actions_exact=True,episode_fast_reset_verified=True,model_and_CPU_RNG_unchanged=True,CPU_queue_replay_fixture=True,all_inputs_are_preexisting_development_training_data=True,no_new_scene_or_closed_loop_rollout=True,actual_robot_policy_rollouts=0,actual_online_actor_updates=0,TTA_utility_proven=False,records=records,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*.py')})
 with(out/'CPU_qualified_direction_bridge_checks.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print('QUALIFIED_TRAINED_DIRECTION_BRIDGE_SOFTWARE_PASS',queries,consumed,accepted,changed,flush=True)
if __name__=='__main__':main()
