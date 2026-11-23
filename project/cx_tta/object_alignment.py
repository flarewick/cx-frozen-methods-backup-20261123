"""Privileged mask-weighted frozen features for OFFLINE diagnosis only."""
import numpy as np


def pool_object(pooled,mask224):
    pooled,mask224=np.asarray(pooled,float),np.asarray(mask224,float)
    if pooled.shape!=(3,4,4,1152) or mask224.shape!=(3,224,224) or not np.isfinite(mask224).all() or np.any(mask224<0):raise ValueError('Invalid object pooling arrays')
    mass=mask224.reshape(3,4,56,4,56).sum(axis=(2,4))
    denom=mass.sum(axis=(1,2));features=np.einsum('cij,cijd->cd',mass,pooled)/np.maximum(denom[:,None],1e-12)
    features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    yy,xx=np.mgrid[:224,:224];geometry=[]
    for mask,d in zip(mask224,denom):
        geometry.extend([float(d>0),d/(224*224),float((mask*xx).sum()/max(d,1e-12)/224),float((mask*yy).sum()/max(d,1e-12)/224)])
    return features.ravel(),np.array(geometry)
