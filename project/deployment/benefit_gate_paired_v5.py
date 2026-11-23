"""Paired-success gate; causal legal features and centered treatment contrast."""
import numpy as np
from benefit_stage_gate import features as compact,CLASSES
from benefit_gate_dynamics_v3 import dynamics
from benefit_gate_latent_v4 import features as latent
DIMS={'compact50':50,'dynamics33':33,'latent_event72':72}
def features(packet,context,family):
 if family=='compact50':return compact(packet,context)
 if family=='dynamics33':return dynamics(packet,context)
 if family=='latent_event72':return latent(packet,context,family)
 raise ValueError('Unknown paired-success feature family')
def contrast(model,x):
 if model['schema']!='paired_success_candidate_gate_v5':raise ValueError('Wrong paired gate')
 z=(np.asarray(x,np.float64)-model['mean'])/model['scale']
 return float(z@model['effect_weight']+model['effect_bias'])
def predict(model,packet,context):
 # Centered outcome model has logits eta-tau/2 and eta+tau/2. Their
 # expected-success difference has tau's sign. These are admission scores,
 # not calibrated probabilities of the four paired outcome classes.
 tau=contrast(model,features(packet,context,model['feature_family']))
 rescue=float(np.exp(-np.logaddexp(0.,-tau)));harm=(1-rescue)/2;denom=rescue+harm
 return dict(zip(CLASSES,(0.,rescue/denom,harm/denom,0.)))
