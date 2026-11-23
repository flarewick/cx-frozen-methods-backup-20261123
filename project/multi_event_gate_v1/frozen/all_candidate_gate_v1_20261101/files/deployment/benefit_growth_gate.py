"""Legal predecision candidate features and learned, risk-weighted gate inference."""
import numpy as np
from cx_tta.onset_observables import validate_packet
from benefit_stage_gate import features as compact_features,CLASSES

def features(packet,context,kind='compact'):
 p=validate_packet(packet)
 compact=compact_features(p,context)
 if kind=='compact':return compact
 if kind!='channels':raise ValueError('Unknown frozen feature family')
 v=p['visual'].astype(np.float64);old=p['past_visual'].astype(np.float64)
 text=p['text'].astype(np.float64);past_text=p['past_text'].astype(np.float64)
 state=p['state'].astype(np.float64);past=p['past_state'].astype(np.float64)
 delta=p['candidate_delta'][0].astype(np.float64)*.5;base=p['base'][0].astype(np.float64)
 values=[compact,v.mean(axis=(0,2,3)).ravel(),v.std(axis=(0,2,3)).ravel(),(v-old).mean(axis=(0,2,3)).ravel(),text.mean(axis=(0,1)),(text-past_text).mean(axis=(0,1)),state.ravel(),(state-past).ravel(),p['ee_pose'].astype(np.float64).ravel(),(p['ee_pose']-p['past_ee_pose']).astype(np.float64).ravel(),base.mean(0),base.std(0),delta.mean(0),delta.std(0),delta.min(0),delta.max(0)]
 x=np.concatenate(values)
 if not np.isfinite(x).all():raise ValueError('Nonfinite legal feature')
 return x

def probabilities(model,packet,context):
 if model.get('schema')!='candidate_utility_gate_v2':raise ValueError('Wrong gate model schema')
 x=features(packet,context,model['feature_family'])
 mean=np.asarray(model['mean'],np.float64);scale=np.asarray(model['scale'],np.float64)
 if x.shape!=mean.shape or np.any(scale<=0):raise ValueError('Gate normalization mismatch')
 xx=(x-mean)/scale
 if model['method']=='softmax':
  logits=xx@np.asarray(model['weight'],np.float64)+np.asarray(model['bias'],np.float64)
  prob=np.exp(logits-logits.max());prob/=prob.sum()
 elif model['method']=='kernel_ridge':
  train=np.asarray(model['train_x'],np.float64)
  kernel=np.exp(-model['gamma']*((train-xx)**2).mean(1))
  prob=np.clip(kernel@np.asarray(model['coefficient'],np.float64),0,1)
  if prob.sum()<1e-12:prob=np.asarray(model['class_prior'],np.float64)
  prob/=prob.sum()
 else:raise ValueError('Unknown learned gate method')
 if not np.isfinite(prob).all():raise ValueError('Nonfinite learned gate scores')
 return prob

def predict(model,packet,context):
 # Original LocalController uses rescue - 2*harm > 0. Represent a frozen
 # selected cost c by positive risk weighting and renormalization. These are
 # explicitly utility-weighted admission scores, not calibrated probabilities.
 prob=probabilities(model,packet,context)
 score=prob.copy();score[2]*=float(model['harm_cost'])/2
 score/=score.sum()
 return dict(zip(CLASSES,map(float,score)))
