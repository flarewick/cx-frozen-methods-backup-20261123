"""Causal inference for the frozen six-model ensemble; no labels or filesystem outcomes."""
import numpy as np
from benefit_followup_gate_v2 import features as original_features,predict as fallback
from benefit_raw_features_TRAIN472_v108 import features as raw_features
from benefit_stage_gate import CLASSES
FAMILIES=['latent_event72','dynamics33','observation488']
def vector(packet,context):return np.r_[original_features(packet,context,'latent_event72'),original_features(packet,context,'dynamics33'),raw_features(packet,context,'observation488')]
def effect(model,packet,context):
 assert model['schema']=='fixed_six_jackknife_effect_v118'and model['feature_families']==FAMILIES and model['margin_multiplier']==1.
 x=vector(packet,context);w=np.asarray(model['raw_member_weights'],np.float64);b=np.asarray(model['raw_member_biases'],np.float64);assert x.shape==(593,)and w.shape==(12,593)and b.shape==(12,);values=x@w.T+b;tau=float(values.mean()+values.std(ddof=0));assert np.isfinite(tau);return tau

def predict(model,packet,context):
 if model['schema']!='fixed_six_jackknife_effect_v118':return fallback(model,packet,context)
 tau=effect(model,packet,context);s=float(np.exp(-np.logaddexp(0.,-tau)));q=np.array([0.,s,(1-s)/2,0.],np.float64);q/=q.sum();assert np.isfinite(q).all();return dict(zip(CLASSES,map(float,q)))
