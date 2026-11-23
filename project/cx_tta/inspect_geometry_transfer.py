"""Frozen-model gradient diagnostics; no optimizer, policy, router or TTA updates."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .paths import write_json
from .audit import digest
from .probe_candidates import read, verify
from .geometry_readout import GeometryReadout, load


def inspect(run):
    p, q, raw, labels = load(run); result = read(run/'learning_result.json')
    assert read(run/'learning_audit.json')['passed'] and result['status']=='complete'
    x={k:torch.from_numpy(v) for k,v in raw.items()}
    y=torch.from_numpy(labels['regression']);valid=torch.from_numpy(labels['regression_valid'])
    records=[]; sources={str(run/f):digest(run/f) for f in ('learning_protocol.json','learning_result.json','learning_audit.json')}
    sources[str(Path(__file__))]=digest(Path(__file__))
    for report in [r for r in result['reports'] if r['condition']=='expert_plus_omit']:
        job=next(j for j in q['jobs'] if j['scene']==report['scene'] and j['condition']==report['condition'])
        assert digest(Path(report['checkpoint']))==report['checkpoint_sha256']
        model=GeometryReadout(p['channel_dim'],job['mean'])
        model.load_state_dict(torch.load(report['checkpoint'],map_location='cpu',weights_only=True))
        params=tuple(v for name,v in model.named_parameters() if not name.startswith('head.'))
        ew=np.array(q['expert_weights']);rw=np.array(q['recovery_weights']);rw[job['held']]=0
        hw=np.zeros(len(y));hw[job['held']]=job['held_weights']
        components={'expert_position':(ew,slice(0,12)), 'expert_rotation':(ew,slice(12,24)),
                    'recovery_train_rotation':(rw,slice(12,24)), 'held_rotation_diagnostic':(hw,slice(12,18))}
        grads={}; norms={}; values={}
        for name,(weights,columns) in components.items():
            pred=model(x)['regression'];mask=torch.zeros_like(valid);mask[:,columns]=valid[:,columns]
            # Keep original valid denominator; no position-weight inflation when removing pose terms.
            per=((pred-y).square()*mask).sum(1)/valid.sum(1).clamp_min(1)
            loss=(per*torch.tensor(weights,dtype=torch.float32)).sum()
            gradient=torch.autograd.grad(loss,params,allow_unused=True)
            flat=torch.cat([torch.zeros_like(v).flatten() if g is None else g.flatten() for v,g in zip(params,gradient)]).detach().double()
            assert torch.isfinite(flat).all()
            grads[name]=flat;norms[name]=float(flat.norm());values[name]=float(loss.detach())
        cosines={}
        for a,b in (('expert_position','held_rotation_diagnostic'),('expert_rotation','held_rotation_diagnostic'),
                    ('expert_position','recovery_train_rotation'),('expert_rotation','recovery_train_rotation')):
            denominator=norms[a]*norms[b]
            cosines[a+'__'+b]=None if denominator<1e-12 else float(grads[a]@grads[b]/denominator)
        records.append(dict(scene=report['scene'],seed=report['seed'],losses=values,shared_gradient_norms=norms,cosines=cosines))
        sources[report['checkpoint']]=report['checkpoint_sha256']
    verify(dict(source_sha256=sources));verify(q);verify(p)
    write_json(run/'transfer_gradient_diagnostic.json',dict(status='complete_readonly_diagnostic',models=9,reports=records,
        scope='Shared attention/trunk gradients, excluding output head. Held labels used only for developer diagnostics, never fitting or model selection. Optimizer steps=0.',
        limitation='Static masked-MSE gradient cosines are not proof of causation, AdamW finite-step behavior, task-success gain, or calibrated uncertainty. Need a matched training ablation before causal claims.',
        source_sha256=sources))
    print('TRANSFER_GRADIENT_DIAGNOSTIC_COMPLETE',records,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);a=parser.parse_args()
    torch.set_num_threads(2);threadpool_limits(2);inspect(a.run.resolve())
