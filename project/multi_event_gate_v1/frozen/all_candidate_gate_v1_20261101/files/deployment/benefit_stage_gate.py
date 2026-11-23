"""Fixed compact causal features and NumPy inference. No metadata/GT inputs."""
import json
import numpy as np
from cx_tta.onset_observables import validate_packet
from cx_tta.structured_action_adapter import ARM_INDICES
CLASSES=('both_fail','rescue','harm','both_success')
def features(packet,context):
 p=validate_packet(packet);ctx=np.asarray(context,np.float32)
 if ctx.shape!=(8,)or not np.isfinite(ctx).all():raise ValueError('Legal causal event context only')
 s=p['state'][0].astype(np.float64);base=p['base'][0].astype(np.float64);delta=p['candidate_delta'][0].astype(np.float64)*.5
 old=p['past_state'][0].astype(np.float64);out=[]
 for arm in (list(range(6)),list(range(7,13))):
  v=s[arm];out.extend([v.mean(),v.std(),v.min(),v.max()])
  gap=base[:,arm]/np.pi-v;out.extend([np.sqrt(np.mean(gap**2)),np.max(np.abs(gap)),gap.mean()])
  d=delta[:,arm];out.extend([np.sqrt(np.mean(d**2)),d.mean(),np.abs(d).max()])
  out.append(np.sqrt(np.mean((v-old[arm])**2)))
 out.extend(s[[6,13]]);out.extend((base[:,[6,13]].mean(0)-s[[6,13]]));out.extend(s[[6,13]]-old[[6,13]])
 for key in ('visual','text'):
  a=p[key].astype(np.float64);out.extend([a.mean(),a.std()])
 out.append(float(np.linalg.norm(p['visual']-p['past_visual'])/(np.linalg.norm(p['past_visual'])+1e-6)))
 ee=p['ee_pose'][0].astype(np.float64);out.extend(ee[:,:3].ravel());out.extend(np.linalg.norm(ee-p['past_ee_pose'][0],axis=1))
 out.extend(p['has_history'].ravel());out.extend(ctx)
 x=np.asarray(out,np.float64)
 if not np.isfinite(x).all():raise ValueError('Nonfinite gate feature')
 return x

def predict(model,packet,context):
 x=features(packet,context)
 mean=np.asarray(model['mean'],np.float64);scale=np.asarray(model['scale'],np.float64)
 if x.shape!=mean.shape:raise ValueError('Gate feature shape changed')
 if model['fail_closed_no_train_rescue']:return dict(zip(CLASSES,(1.,0.,0.,0.)))
 logits=((x-mean)/scale)@np.asarray(model['weight'],np.float64)+np.asarray(model['bias'],np.float64)
 prob=np.exp(logits-logits.max());prob/=prob.sum()
 return dict(zip(CLASSES,map(float,prob)))
