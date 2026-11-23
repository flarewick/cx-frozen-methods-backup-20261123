"""Candidate checkpoint verification without importing the live policy stack."""
import json
from pathlib import Path
import torch
from candidate_eval_reference import sha
D=Path(__file__).resolve().parent;R=D.parent;STUDY=R/'multi_event_gate_v1/outputs/training_diagnostics/shared6_vs_single_20261025'
CANDIDATES={'M_shared250_a050':dict(job='shared6_lambda0p0',step=250,alpha=.5),
            'M_single100_a025':dict(job='open_laptop_lambda0p0',step=100,alpha=.25)}
PAYLOADS={}

def candidate(condition):
    if condition in PAYLOADS:return PAYLOADS[condition]
    spec=CANDIDATES[condition];folder=STUDY/spec['job'];path=folder/'best_scaled.pt'
    result=json.loads((folder/'result.json').read_text());audit=json.loads((STUDY/'audit.json').read_text());protocol=STUDY/'protocol.json'
    if not audit['passed'] or not audit['all9complete'] or audit['protocol_sha256']!=sha(protocol) or result['protocol_sha256']!=sha(protocol):raise RuntimeError('Diagnostic candidate audit/identity invalid')
    if result['checkpoint_sha256']['best_scaled.pt']!=sha(path):raise RuntimeError('Candidate checkpoint SHA differs')
    payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata'];selection=meta['selection']
    if meta['scope']!='offline_train_validation_diagnostic_only' or meta['protocol_sha256']!=sha(protocol) or meta['job']!=result['job'] or selection!=result['best_scaled'] or selection['step']!=spec['step'] or selection['alpha']!=spec['alpha']:raise RuntimeError('Wrong selected checkpoint or scale')
    if any(not torch.isfinite(v).all() for v in payload['model'].values()):raise RuntimeError('Nonfinite candidate state')
    PAYLOADS[condition]=(payload,path,spec)
    return PAYLOADS[condition]

