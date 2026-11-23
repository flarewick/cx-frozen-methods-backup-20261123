"""Freeze the single predeclared auxiliary predictor after complete developer audit."""
from pathlib import Path
import importlib.util,numpy as np
C=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('actual_phase_feedback_auxiliary_freeze',C/'study.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
def run():
 audit=s.read(s.D/'independent_audit.json');report=s.read(s.D/'report.json')
 assert audit['passed']and audit['developer_auxiliary_support_gate_passed']and report['developer_auxiliary_support_gate_passed']
 p,x,a,meta,eligible,allocation=s.dataset();ids=np.flatnonzero(eligible)
 xx,v,kind,gate,donors=s.old.inputs(x,a,meta,ids,ids,'correct');assert not donors
 state,scales=s.old.fit(xx[ids],v[ids],kind[ids],a['target'][ids],[meta[i]for i in ids])
 out=s.O/'frozen_auxiliary';out.mkdir(parents=True,exist_ok=True);file=out/'predictor.npz'
 with file.open('xb')as f:np.savez_compressed(f,**state)
 frozen=s.load(file);expected=s.m.predict(state,xx[ids],v[ids],kind[ids]);actual=s.m.predict(frozen,xx[ids],v[ids],kind[ids])
 assert np.array_equal(actual,expected)and np.isfinite(actual).all()
 assert np.array_equal(s.m.predict(frozen,xx[ids[:32]],np.zeros((32,194)),kind[ids[:32]]),np.zeros((32,2)))
 sources=[C/n for n in('freeze_model.py','study.py','feedback_queue.py','eligibility.py','replay.py','checks.py','audit.py')]+[s.T/'native_local_response_v2/model.py',s.T/'native_local_response_v2/study.py']
 protocol=dict(auxiliary_candidate_frozen=True,FIT_rows=len(ids),ridge=1.,primary_model='correct',
  checkpoint_sha256=s.sha(file),train_target_scales=scales,
  model_inputs='Current causal context53, actualcommand197 transformed to object-local194 + ownkind5, currentRGB corner cloud24x2. No task/seed/GT/future.',
  response='Signed sameID local 2D material displacement; no object-pose/contact/task-success claim.',
  online_timing='Fulloriginal32 + actual95; window starts>=95 and ends>=95+.25; once per source only at actual caller query boundary. q2 plans not yet observed.',
  actual_recorded_q1_packet_rows=len(allocation['actual_q1_rows']),NOT_yet_recorded_q2_packet_rows=len(allocation['hypothetical_q2_rows']),
  full_online_candidate_frozen=False,shared_fast128_response_interface_integrated=False,
  new_independent_confirmation_passed=False,direction_training_allowed=False,
  source_sha256={str(p):s.sha(p)for p in sources},
  input_sha256={str(s.D/n):s.sha(s.D/n)for n in('protocol.json','report.json','independent_audit.json','all_models_frozen.json')},
  checkpoint_reload_predictions_bitwise=True,physical_null_exact=True,
  online_updates=0,TTA_utility_proven=False)
 s.write(out/'protocol.json',protocol);print('SINGLE_PHASE_AUXILIARY_CANDIDATE_FROZEN',len(ids),file,s.sha(file),flush=True)
if __name__=='__main__':run()
