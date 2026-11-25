from pathlib import Path
import torch
from .common import C,W,T,read,sha
from native_action_visible_response_v1.model import ActionVisibleResponse
from native_shared_response_v1.session import state_sha
O=C/'pi05_gt_shared6_v1'
def qualified(model,file):
 file=Path(file).resolve();assert file.is_relative_to(O.resolve());q=read(file)
 assert q['frozen']and q['trained_response_only']and q['actual_updates']==1000 and q['execution_alpha']==.5
 assert q['model_kind']==ActionVisibleResponse.model_kind and type(model)is ActionVisibleResponse
 for name,h in q['source_sha256'].items():assert sha(name)==h
 for name,h in q['input_sha256'].items():assert sha(name)==h
 assert read(q['actual_exit'])['actual_returncode']==0
 assert q['shared6_adapter_sha256']==sha(W/'outputs/training_diagnostics/shared6_vs_single_20261025/shared6_lambda0p0/best_scaled.pt')
 assert q['v118_gate_sha256']==sha(W/'outputs/benefit_followup_v2/followup_20261027_v2/constant_owner_v4/fixed_ensemble_TRAIN472_v118/diagnostic_gate.json')
 assert state_sha(model.response.state_dict())==q['response_state_sha256']
 expected=torch.load(q['checkpoint'],map_location='cpu',weights_only=True)
 assert set(expected['buffers'])==set(dict(model.named_buffers()))
 for k,v in model.named_buffers():assert torch.equal(v,expected['buffers'][k]),k
 return sha(file)
def load_model(file):
 q=read(file);basis=torch.load(q['basis'],map_location='cpu',weights_only=True)
 model=ActionVisibleResponse(*[basis[k]for k in ['arm_weight','context_mean','context_scale','action_scale']]).eval()
 payload=torch.load(q['checkpoint'],map_location='cpu',weights_only=True);assert payload['step']==1000
 model.response.load_state_dict(payload['response'],strict=True);qualified(model,file);return model,q
