"""Target-free confidence proxies; deliberately no masks or metadata argument."""
import numpy as np


def scores(context,base,proposals,reference_context):
    if proposals.shape[0]!=3 or proposals.shape[1:]!=base.shape:raise ValueError('Three frozen initialization models required')
    # All32 predicted commands are known at decision time. Target-valid tails
    # must NOT change the confidence score or reveal the teacher horizon.
    delta=(proposals.astype(float)-base[None]).reshape(3,len(base),32,2,7)
    x=context.astype(float);ref=reference_context.astype(float)
    distance=np.maximum((x*x).sum(1)[:,None]+(ref*ref).sum(1)[None]-2*x@ref.T,0)
    novelty=np.sqrt(distance.min(1)/x.shape[1])[:,None].repeat(2,1)
    result={}
    for component,z in [('arm',delta[...,:6]/.1),('grip',delta[...,6:]/.2)]:
        spread=np.sqrt(np.mean(np.var(z,axis=0),axis=(1,3)))
        amplitude=np.sqrt(np.mean(np.mean(z,axis=0)**2,axis=(1,3)))
        result[component]=dict(low_disagreement=-spread,relative_agreement=-spread/(amplitude+1e-6),
            low_correction_magnitude=-amplitude,near_fit_context=-novelty)
    return result


def weighted_auc(score,label,weights):
    pos=float(weights[label].sum());neg=float(weights[~label].sum())
    if pos==0 or neg==0:return None
    _,bins=np.unique(score,return_inverse=True)
    p=np.bincount(bins,weights=weights*label);n=np.bincount(bins,weights=weights*(~label))
    return float(np.sum(p*(np.cumsum(n)-n+.5*n))/(pos*neg))


def rank_stats(score,label,weights):
    prevalence=float(np.average(label,weights=weights));threshold=float(np.quantile(score,.9))
    chosen=score>=threshold;coverage=float(weights[chosen].sum()/weights.sum())
    return dict(auc=weighted_auc(score,label,weights),label_prevalence=prevalence,
        top10_rank_threshold=threshold,actual_weighted_coverage=coverage,
        top10_weighted_precision=float(np.average(label[chosen],weights=weights[chosen])),
        selected_units=int(chosen.sum()),selected_positive_units=int((chosen&label).sum()),
        positive_units=int(label.sum()),units=len(label),
        positive_recall=None if not weights[label].sum() else float(weights[chosen&label].sum()/weights[label].sum()))
