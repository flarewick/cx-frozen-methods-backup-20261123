"""Small CPU-only four-outcome readout prototype; no runtime gate integration.

All scaling/PCA/weights use fit scenes only. Calibration packets are never read
by fit_readout. Predictions are uncalibrated scores, NOT probability guarantees
or BenefitRiskBounds. A future frozen recipe and independent validation remain
necessary before using these outputs for any intervention.
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np
import torch
from .onset_decision_dataset import CLASSES, require_fit_support
from .onset_router_features import feature_groups

OBS_GROUPS=('observation','observed_change','base_plan')
FULL_GROUPS=OBS_GROUPS+('candidate','candidate_interaction')


@dataclass
class GroupProjection:
    mean: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    projected_scale: np.ndarray

    def transform(self, values):
        return ((values-self.mean)/self.scale)@self.basis/self.projected_scale


@dataclass
class CandidateBenefitReadout:
    projections: dict
    weights: np.ndarray
    bias: np.ndarray
    metadata: dict

    def encode(self, packet):
        fields=feature_groups(packet)
        values=np.concatenate([p.transform(fields[k].astype(np.float64))
                               for k,p in self.projections.items()]).astype(np.float32)
        if not np.isfinite(values).all():raise ValueError('Nonfinite encoded candidate features')
        return values

    def predict(self, packet):
        z=self.encode(packet)@self.weights.T+self.bias
        if not np.isfinite(z).all():raise ValueError('Nonfinite candidate scores')
        scores=np.exp(z-z.max());scores/=scores.sum()
        return dict(class_scores={k:float(v) for k,v in zip(CLASSES,scores)},
            rescue_score=float(scores[1]),harm_score=float(scores[2]),
            incremental_score=float(scores[1]-scores[2]),calibrated=False,
            deployment_approved=False)


def fit_readout(packets, rows, criteria, *, include_candidate=True, rank=4,
                steps=200, learning_rate=.03, l2=.1):
    """Prototype recipe; callers must freeze settings/data before real fitting.

    Whole-scene support is mandatory. Returned model never implies intervention
    approval. With rank4 each of five groups has <=4 components, so the shared
    linear four-class head has at most84 trainable parameters.
    """
    require_fit_support(rows,criteria)
    if len(packets)!=len(rows) or not all(r['index']==i for i,r in enumerate(rows)):
        raise ValueError('One indexed packet per offline decision row required')
    if (type(include_candidate)is not bool or type(rank)is not int or not 1<=rank<=4 or type(steps)is not int or steps<1
            or not np.isfinite([learning_rate,l2]).all() or learning_rate<=0 or l2<=0):
        raise ValueError('Bounded rank, positive budget and regularization required')
    fit=[i for i,r in enumerate(rows) if r['split']=='fit']
    # Do not even access calibration packet values here.
    fields=[feature_groups(packets[i]) for i in fit]
    names=FULL_GROUPS if include_candidate else OBS_GROUPS
    projections={};encoded=[]
    for name in names:
        v=np.stack([f[name] for f in fields]).astype(np.float64)
        mean=v.mean(0);scale=np.maximum(v.std(0),1e-3)
        standardized=(v-mean)/scale
        _,singular,vh=np.linalg.svd(standardized,full_matrices=False)
        k=min(rank,int(np.count_nonzero(singular>1e-7)))
        if k==0:continue  # Constant observable group has no fitting signal.
        basis=vh[:k].T.copy()
        # Fix the sign ambiguity for reproducible saved projections.
        for j in range(k):
            if basis[np.abs(basis[:,j]).argmax(),j]<0:basis[:,j]*=-1
        projected=standardized@basis
        p=GroupProjection(mean,scale,basis,np.maximum(projected.std(0),1e-3))
        projections[name]=p;encoded.append(p.transform(v))
    if not encoded:raise ValueError('No observable variation in fit scenes')
    x=torch.tensor(np.concatenate(encoded,axis=1),dtype=torch.float32)
    y=torch.tensor([CLASSES.index(rows[i]['label']) for i in fit],dtype=torch.long)
    counts=Counter(rows[i]['group'] for i in fit)
    w=torch.tensor([1/counts[rows[i]['group']] for i in fit],dtype=torch.float32);w/=w.sum()
    # Same exact zero initialization, CPU full batches and budget across ablations.
    weights=torch.zeros((4,x.shape[1]),requires_grad=True)
    bias=torch.zeros(4,requires_grad=True)
    optimizer=torch.optim.Adam([weights,bias],lr=learning_rate)
    losses=[]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits=x@weights.T+bias
        loss=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*w).sum()+l2*weights.square().sum()/2
        if not torch.isfinite(loss):raise ValueError('Nonfinite readout fit')
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    return CandidateBenefitReadout(projections,weights.detach().numpy().copy(),bias.detach().numpy().copy(),
        dict(classes=CLASSES,fit_indices=fit,fit_scenes=sorted(counts),rank_per_group=rank,
             include_candidate=include_candidate,steps=steps,learning_rate=learning_rate,l2=l2,
             head_parameters=weights.numel()+bias.numel(),initialization='exact_zero',
             first_loss=losses[0],last_loss=losses[-1],normalization_PCA_fit_only=True,
             no_calibration_packets_read=True,calibrated=False,deployment_approved=False))
