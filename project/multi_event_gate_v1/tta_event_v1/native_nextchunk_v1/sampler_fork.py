"""In-memory fresh sampler fork. No archived RNG import, no environment restore."""
from contextlib import contextmanager
import copy
import hashlib
import os
import pickle
import uuid
import numpy as np
import torch
from query_ledger import QueryLedger

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()

def environment_hash(state):
    return sha_bytes(pickle.dumps((state['numpy'],state['python']),protocol=5))

def torch_hash(state):
    h=hashlib.sha256()
    for tensor in (state['torch_cpu'],*state['torch_cuda']):
        if tensor.device.type!='cpu' or tensor.dtype!=torch.uint8:
            raise ValueError('Torch RNG states must be original CPU ByteTensors')
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()

def full_hash(state):
    return sha_bytes((environment_hash(state)+torch_hash(state)).encode())

def base_hash(base):
    base=np.asarray(base)
    if base.dtype!=np.float32 or base.shape!=(32,14) or not np.isfinite(base).all():
        raise ValueError('Exact finite original float32 base32x14 required')
    return sha_bytes(base.tobytes())

class NoiseWitness:
    """Observe the real sample_noise call; never supply or replace its values."""
    def __init__(self,model,router,lane):
        self.model,self.router,self.lane=model,router,lane
        self.calls=[]
    @contextmanager
    def installed(self):
        if self.router.current() is not self.lane:
            raise ValueError('Sampler must execute in its actual owner lane')
        had='sample_noise' in self.model.__dict__
        previous=self.model.__dict__.get('sample_noise')
        original=self.model.sample_noise
        def observed(shape,device):
            if self.calls:raise ValueError('More than one internal-noise draw in a query')
            if tuple(shape)!=(1,32,32):raise ValueError('Frozen sampler noise shape changed')
            selected=torch.device(device)
            if selected.type!=('cuda' if self.router.cuda else 'cpu'):
                raise ValueError('Noise escaped the selected single device')
            before=self.router.capture(self.lane)
            value=original(shape,device)
            after=self.router.capture(self.lane)
            if value.shape!=(1,32,32) or value.dtype!=torch.float32 or value.requires_grad:
                raise ValueError('Original noise tensor contract changed')
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite internal noise')
            self.calls.append(dict(
                before=torch_hash(before),after=torch_hash(after),
                environment_before=environment_hash(before),
                environment_after=environment_hash(after),
                noise_sha256=sha_bytes(value.detach().cpu().contiguous().numpy().tobytes()),
                shape=[1,32,32],device_type=selected.type))
            return value
        self.model.sample_noise=observed
        try:yield self
        finally:
            if had:self.model.sample_noise=previous
            else:delattr(self.model,'sample_noise')

def audited_draw(router,lane,model,ledger,operation,*,query,tick):
    """operation owns the actual infer(noise=None); fail closed after any bad draw."""
    ledger.permit_query(query,tick)
    before=router.capture(lane)
    witness=NoiseWitness(model,router,lane)
    with witness.installed():
        result=operation()
    after=router.capture(lane)
    if len(witness.calls)!=1:
        raise ValueError('Query did not consume exactly one real internal-noise draw')
    note=witness.calls[0]
    if environment_hash(before)!=environment_hash(after):
        raise ValueError('Query advanced Python/NumPy environment RNG')
    if (note['before']!=torch_hash(before) or note['after']!=torch_hash(after)
            or note['before']==note['after']
            or note['environment_before']!=note['environment_after']):
        raise ValueError('Torch RNG advanced outside the single witnessed noise call')
    digest=base_hash(result['base'])
    ledger.record_draw(query,tick,digest,full_hash(before),full_hash(after),
                       actual_sampler_call=True)
    return result,copy.deepcopy(after),dict(query=query,tick=tick,base_sha256=digest,
        rng_before=full_hash(before),rng_after=full_hash(after),
        environment_unchanged=True,**note)

class FreshSamplerFork:
    """Fresh discovery stream and branch clones, valid in this process only."""
    def __init__(self,router,lane,model,*,device_uuid,model_sha256):
        if (router.cuda and not device_uuid.startswith('GPU-')) or (not router.cuda and device_uuid!='CPU'):
            raise ValueError('Exact current device identity required')
        if len(model_sha256)!=64:raise ValueError('Frozen model SHA required')
        self.router,self.lane,self.model=router,lane,model
        self.device_uuid,self.model_sha256=device_uuid,model_sha256
        self.owner_pid=os.getpid()
        self.nonce=uuid.uuid4().hex
        self.ledger=QueryLedger('fresh_discovery')
        self.notes=[]
        self._after=None
        self.poisoned=False
        self._branches=set()
    def _identity(self,router,*,device_uuid,model_sha256):
        if self.poisoned or self.owner_pid!=os.getpid():
            raise ValueError('Invalid/foreign-process sampler snapshot')
        if router is not self.router or device_uuid!=self.device_uuid or model_sha256!=self.model_sha256:
            raise ValueError('Sampler router/device/model identity changed')
    def draw(self,operation,*,query,tick):
        self._identity(self.router,device_uuid=self.device_uuid,model_sha256=self.model_sha256)
        try:
            result,self._after,note=audited_draw(self.router,self.lane,self.model,
                self.ledger,operation,query=query,tick=tick)
        except BaseException:
            self.poisoned=True
            raise
        self.notes.append(note)
        return result
    def finish_prefix(self,returned,*,end_tick):
        if self.poisoned:raise ValueError('Poisoned discovery')
        self.ledger.finish_chunk(returned,end_tick=end_tick)
    def branch(self,name,router,lane,*,device_uuid,model_sha256):
        self._identity(router,device_uuid=device_uuid,model_sha256=model_sha256)
        if name in self._branches or not name or not self.ledger.awaiting_execution or self._after is None:
            raise ValueError('Unique branch at the fresh selected pre-execution query required')
        self._branches.add(name)
        return BranchFork(self,name,lane)

class BranchFork:
    def __init__(self,source,name,lane):
        self.source,self.name,self.lane=source,name,lane
        self.ledger=copy.deepcopy(source.ledger)
        self.ledger.branch=name
        self._snapshot=copy.deepcopy(source._after)
        self._snapshot_hash=torch_hash(self._snapshot)
        self._legacy_hash=None
        self.installed=False
        self.poisoned=False
        self.following_draws=0
        self.handoff=None
    def _identity(self,device_uuid,model_sha256):
        self.source._identity(self.source.router,device_uuid=device_uuid,model_sha256=model_sha256)
        if self.poisoned or self.source.router.current() is not self.lane:
            raise ValueError('Branch is poisoned or outside its owner lane')
    def remember_legacy_boundary(self,*,device_uuid,model_sha256):
        self._identity(device_uuid,model_sha256)
        if self._legacy_hash is not None or self.installed:
            raise ValueError('Legacy boundary may be recorded only once before original execution')
        self._legacy_hash=torch_hash(self.source.router.capture(self.lane))
    def finish_origin(self,returned,*,end_tick,requested95_arrived):
        if self.poisoned:raise ValueError('Poisoned branch')
        self.ledger.finish_chunk(returned,end_tick=end_tick,
             requested95_arrived=requested95_arrived,origin=True)
    def install(self,*,tick,device_uuid,model_sha256):
        self._identity(device_uuid,model_sha256)
        self.ledger.require_following_query(tick)
        if self.installed or self._legacy_hash is None:
            raise ValueError('One explicit legacy-to-sampler handoff required')
        before=self.source.router.capture(self.lane)
        if torch_hash(before)!=self._legacy_hash:
            raise ValueError('Unknown Torch draws during native prefix/origin; do not overwrite')
        if torch_hash(self._snapshot)!=self._snapshot_hash:
            raise ValueError('Fresh in-memory sampler snapshot was mutated')
        if len(self._snapshot['torch_cuda'])!=(1 if self.source.router.cuda else 0):
            raise ValueError('Exactly one selected CUDA RNG stream, or CPU test, required')
        torch.set_rng_state(self._snapshot['torch_cpu'].clone())
        if self.source.router.cuda:
            torch.cuda.set_rng_state_all([x.clone() for x in self._snapshot['torch_cuda']])
        after=self.source.router.capture(self.lane)
        if environment_hash(before)!=environment_hash(after) or torch_hash(after)!=self._snapshot_hash:
            self.poisoned=True
            raise ValueError('Torch-only handoff changed environment or lost sampler state')
        self.installed=True
        self.handoff=dict(branch=self.name,owner_pid=self.source.owner_pid,nonce=self.source.nonce,
             device_uuid=device_uuid,model_sha256=model_sha256,
             query_count=len(self.ledger.records),base_sha256=self.ledger.records[-1]['base_sha256'],
             tick=tick,environment_unchanged=True,
             legacy_torch_hash=torch_hash(before),fresh_sampler_torch_hash=torch_hash(after),
             legacy_combined_rng_not_claimed_equal_to_after_handoff=True)
        return copy.deepcopy(self.handoff)
    def draw_following(self,operation,*,query,tick,device_uuid,model_sha256):
        self._identity(device_uuid,model_sha256)
        if not self.installed or self.following_draws>=1:
            raise ValueError('Exactly one following natural query in this protocol')
        self.ledger.require_following_query(tick)
        try:
            result,after,note=audited_draw(self.source.router,self.lane,self.source.model,
                   self.ledger,operation,query=query,tick=tick)
        except BaseException:
            self.poisoned=True
            raise
        self.following_draws+=1
        return result,note
    def finish_following(self,returned,*,end_tick):
        if self.following_draws!=1:raise ValueError('No following query to finish')
        self.ledger.finish_chunk(returned,end_tick=end_tick)
