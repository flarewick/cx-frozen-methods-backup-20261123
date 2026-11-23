"""Frozen adapter encoding of the same causal packet; no GT/model retraining."""
import numpy as np,torch
from benefit_growth_protocol import CANDIDATE,sha
from cx_tta.onset_observables import validate_packet
from cx_tta.structured_action_adapter import FORWARD_KEYS
from benefit_gate_dynamics_v3 import dynamics
EXPECTED='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7'
_MODEL=None

def encoder():
 global _MODEL
 if _MODEL is None:
  if sha(CANDIDATE)!=EXPECTED:raise RuntimeError('Frozen adapter checkpoint changed')
  from task_adapters.adapter import StackingAdapter
  payload=torch.load(CANDIDATE,map_location='cpu',weights_only=True)
  # Constructor initialization must not perturb the policy/sampler RNG.
  with torch.random.fork_rng(devices=[]):
   m=StackingAdapter(32,payload['model']['reg_mean'].tolist());m.load_state_dict(payload['model'],strict=True)
  m.eval();m.requires_grad_(False);_MODEL=m
 return _MODEL

def encode(packet):
 p=validate_packet(packet);model=encoder()
 x={k:torch.from_numpy(p[k].copy())for k in FORWARD_KEYS}
 # Exact current-only mask and concatenation from StackingAdapter.forward and
 # StructuredActionAdapter.forward. Original-forward equality is CPU audited.
 current={k:torch.zeros_like(v)if k.startswith('past_')or k=='has_history'else v for k,v in x.items()}
 with torch.inference_mode():
  cur,cw=model.attention(current,'','conditional');past,pw=model.attention(current,'past_','conditional')
  merged=torch.cat([cur,current['text'].flatten(1),past,current['past_text'].flatten(1),current['state'],current['past_state'],current['ee_pose'].flatten(1),current['past_ee_pose'].flatten(1),current['has_history']],dim=1)
  latent=model.trunk(merged)
 return latent[0].numpy().astype(np.float64),cw[0].numpy().reshape(-1).astype(np.float64)

DIMS={'latent_event72':72,'latent_dynamics97':97,'latent_attention84':84}
def features(packet,context,family):
 p=validate_packet(packet);ctx=np.asarray(context,np.float32)
 if ctx.shape!=(8,)or not np.isfinite(ctx).all():raise ValueError('Eight legal event context values required')
 z,attention=encode(p)
 if family=='latent_event72':x=np.r_[z,ctx]
 elif family=='latent_dynamics97':x=np.r_[z,dynamics(p,ctx)]
 elif family=='latent_attention84':x=np.r_[z,attention,ctx]
 else:raise ValueError('Unknown preregistered encoder feature family')
 if x.shape!=(DIMS[family],)or not np.isfinite(x).all():raise ValueError('Nonfinite causal encoder feature')
 return x
