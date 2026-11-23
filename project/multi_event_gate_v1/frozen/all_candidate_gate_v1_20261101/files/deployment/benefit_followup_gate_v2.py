"""Whitelisted causal predictors for the followup, with a frozen coefficient file."""
import numpy as np
from cx_tta.onset_observables import validate_packet
from benefit_stage_gate import features as compact,CLASSES
from benefit_gate_dynamics_v3 import features as dynamic
from benefit_gate_latent_v4 import features as latent

def features(packet,context,family):
 validate_packet(packet)
 if family=='compact50':return compact(packet,context)
 if family in('dynamics33','dynamics_spatial81','compact_dynamics83'):return dynamic(packet,context,family)
 if family in('latent_event72','latent_dynamics97','latent_attention84'):return latent(packet,context,family)
 raise ValueError('Unknown frozen followup feature family')
def predict(model,packet,context):
 if model['schema']!='followup_candidate_gate_v2':raise ValueError('Wrong followup gate schema')
 x=features(packet,context,model['feature_family']);mean=np.asarray(model['mean'],np.float64);scale=np.asarray(model['scale'],np.float64)
 if x.shape!=mean.shape or np.any(scale<=0):raise ValueError('Normalization mismatch')
 z=(x-mean)/scale
 if model['predictor_kind']=='outcome_softmax':
  logits=z@np.asarray(model['weight'],np.float64)+np.asarray(model['bias'],np.float64);p=np.exp(logits-logits.max());p/=p.sum();p[2]*=.5;p/=p.sum()
 elif model['predictor_kind']=='paired_success':
  tau=float(z@np.asarray(model['effect_weight'],np.float64)+model['effect_bias']);s=float(np.exp(-np.logaddexp(0.,-tau)));h=(1-s)/2;p=np.array([0.,s,h,0.],np.float64);p/=p.sum()
 else:raise ValueError('Unregistered predictor kind')
 if not np.isfinite(p).all():raise ValueError('Nonfinite followup admission scores')
 return dict(zip(CLASSES,map(float,p)))
