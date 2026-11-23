"""Causal risk features from known proposals and native legal observations only."""
import numpy as np
from cx_tta.onset_observables import validate_packet
from benefit_stage_gate import features as compact

def _rms(x):return float(np.sqrt(np.mean(np.square(x))))
def _cos(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel()
 return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def dynamics(packet,context):
 p=validate_packet(packet);ctx=np.asarray(context,dtype=np.float32)
 if ctx.shape!=(8,)or not np.isfinite(ctx).all():raise ValueError('Exactly eight legal context values required')
 state=p['state'][0].astype(np.float64);old=p['past_state'][0].astype(np.float64)
 base=p['base'][0].astype(np.float64);delta=p['candidate_delta'][0].astype(np.float64)*.5
 history=bool(p['has_history'][0,0]);out=[]
 for offset in (0,7):
  ids=list(range(offset,offset+6));s=state[ids]*np.pi;past=old[ids]*np.pi
  b=base[:,ids];d=delta[:,ids];gap=b-s;v=np.diff(b,axis=0);dv=np.diff(d,axis=0)
  grip=base[:,offset+6];change=np.abs(np.diff(np.r_[state[offset+6],grip]))
  coupling=float((np.linalg.norm(d,axis=1)*change).sum()/(change.sum()+1e-8))
  out.extend([_rms(d),float(np.max(np.abs(d))),_cos(d,gap),float(np.log1p(_rms(d)/(_rms(gap)+.01))),_rms(v+dv)-_rms(v),float(np.linalg.norm(gap[0]+d[0])-np.linalg.norm(gap[0])),_rms(np.diff(v+dv,axis=0))-_rms(np.diff(v,axis=0)),coupling,_rms(s-past)if history else 0.,float(np.mean(grip)-state[offset+6]),float(np.mean(grip<state[offset+6]-.05)),float(np.linalg.norm((p['ee_pose'][0,offset//7,:3]-p['past_ee_pose'][0,offset//7,:3])*.1))if history else 0.])
 out.extend(ctx);out.append(float(history));x=np.asarray(out,dtype=np.float64)
 if x.shape!=(33,)or not np.isfinite(x).all():raise ValueError('Finite dynamics33 required')
 return x

def spatial(packet):
 p=validate_packet(packet);v=p['visual'][0].astype(np.float64).reshape(12,32);old=p['past_visual'][0].astype(np.float64).reshape(12,32);text=p['text'][0].astype(np.float64).mean(0)
 history=bool(p['has_history'][0,0]);out=[]
 for a,b in zip(v,old):out.extend([_rms(a),_rms(a-b)if history else 0.,_cos(a,b)if history else 0.,_cos(a,text)])
 x=np.asarray(out,dtype=np.float64)
 if x.shape!=(48,)or not np.isfinite(x).all():raise ValueError('Finite spatial48 required')
 return x

DIMS={'dynamics33':33,'dynamics_spatial81':81,'compact_dynamics83':83}
def features(packet,context,family):
 d=dynamics(packet,context)
 if family=='dynamics33':return d
 if family=='dynamics_spatial81':return np.r_[d,spatial(packet)]
 if family=='compact_dynamics83':return np.r_[compact(packet,context),d]
 raise ValueError('Unknown predeclared causal risk family')
