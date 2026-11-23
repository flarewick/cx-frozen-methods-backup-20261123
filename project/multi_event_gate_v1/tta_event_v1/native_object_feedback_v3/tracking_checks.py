"""Actual immutable SAM2 CPU inference on retained real arrived RGB, no efficacy claim."""
from pathlib import Path
import json,numpy as np,torch,hashlib
from .tracking import MaskCore,video
from query_object_v1.stream import MaskCore as Previous
T=Path(__file__).resolve().parents[1]
def main():
 torch.set_num_threads(2);root=T/'outputs/native_frozen100_v25/mixed/batches/033/runs/99bed8c62b6d463988e7e0e32efe88f7'
 answers=sorted((root/'semantic/answers').glob('*.json'));answer=next(json.loads(p.read_text())for p in answers if json.loads(p.read_text())['available'])
 frame=Path(answer['frame_file']);mask=Path(answer['mask_file'])
 with np.load(frame,allow_pickle=False)as z:rgb=z['cam0'].copy()
 with np.load(mask,allow_pickle=False)as z:m=z['masks'].copy()
 with torch.random.fork_rng(devices=[]):model=video.build('cpu')
 import sam2.sam2_video_predictor as official
 official.tqdm=lambda x,**kw:x
 def run(core,masks):
  with torch.inference_mode():
   a=core.initialize(rgb,masks,tick=0,decision_tick=0);b=core.append(rgb,tick=25,decision_tick=25)
  return a,b
 original=run(Previous(model),m);fresh=run(MaskCore(model),m)
 for a,b in zip(original,fresh):
  for k in ('masks','presence'):assert np.array_equal(a[k],b[k]),k
  assert a['raw_logits_sha256']==b['raw_logits_sha256']
 single=m.copy();single[1]=False;core=MaskCore(model);rows=run(core,single)
 assert core.count==1 and list(core.stream.state['obj_id_to_idx'])==[0]
 for row in rows:assert row['masks'][0].any()and not row['masks'][1].any()and row['presence'][1]==0
 try:core.append(rgb,tick=24,decision_tick=24)
 except ValueError:pass
 else:raise AssertionError('Backward image accepted')
 assert not torch.cuda.is_initialized()
 value=dict(passed=True,actual_frozen_SAM2_CPU_inference=True,actual_two_object_output_exact_previous=True,actual_one_object_SAM2_ids=[0],empty_reference_never_registered=True,backward_frame_rejected=True,actual_real_RGB_source=str(frame),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in(frame,mask)},CUDA_initialized=False,scientific_efficacy_claim=False)
 out=T/'outputs/native_object_feedback_v3/CPU_actual_SAM2_checks.json';out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print(json.dumps(value))
if __name__=='__main__':main()
