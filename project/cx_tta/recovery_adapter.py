"""Observable-only frozen joint-prefix interface for one bounded intervention."""
import time
import numpy as np
import torch
from .cache_joint_prefix import encode
from .dataset import context_features
from .deploy import unpack_observation
from .direction_adapter import DirectionAdapter
from .recovery_contract import rng_capture,rng_fingerprint


class RecoveryAdapter:
    def __init__(self,weights,preprocessing):
        with np.load(preprocessing) as z:self.pre=dict(z)
        self.model=DirectionAdapter(context_dim=len(self.pre['context_mean']),mode='bc').eval()
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.encode_eager=torch.compiler.disable(encode)

    def observable_features(self,policy,observation,instruction):
        images,state=unpack_observation(observation)
        before=rng_fingerprint(rng_capture())
        config=policy._model.paligemma_with_expert.paligemma.language_model.config
        implementation=config._attn_implementation
        started=time.perf_counter()
        try:feature,meta=self.encode_eager(policy,images,state,instruction)
        finally:config._attn_implementation=implementation
        if before!=rng_fingerprint(rng_capture()):raise RuntimeError('Feature extraction changed RNG')
        visual=(feature.astype(np.float32)-self.pre['pca_mean'])@self.pre['pca_components'].T
        current=np.r_[visual,context_features(images,state)[:14]].astype(np.float32)
        return current,dict(**meta,seconds=time.perf_counter()-started,rng_unchanged=True)

    def propose(self,current,previous,base):
        available=previous is not None
        context=np.r_[current,previous if available else np.zeros_like(current),float(available)].astype(np.float32)
        context=((context-self.pre['context_mean'])/self.pre['context_std']).astype(np.float32)
        with torch.inference_mode():_,delta,_=self.model(torch.from_numpy(context)[None],torch.from_numpy(base)[None])
        delta=delta[0].numpy().copy()
        if not np.isfinite(delta).all() or np.any(np.abs(delta)>self.model.caps.numpy()+1e-6):raise ValueError('Invalid residual')
        return delta,context
