"""Fresh common-prefix fork for a CURRENT-RGB, normally continued policy.
Installs the already-consumed sampler state BEFORE the selected original chunk.
Legacy end-of-origin handoff remains separately preserved in sampler_fork.py.
"""
import copy
import torch
from sampler_fork import FreshSamplerFork,BranchFork,torch_hash,environment_hash

class CurrentQueryFork(FreshSamplerFork):
    def branch(self,name,router,lane,*,device_uuid,model_sha256):
        self._identity(router,device_uuid=device_uuid,model_sha256=model_sha256)
        if name in self._branches or not name or not self.ledger.awaiting_execution or self._after is None:
            raise ValueError('Unique fresh selected-query branch required')
        self._branches.add(name)
        return CurrentBranch(self,name,lane)

class CurrentBranch(BranchFork):
    def install(self,**kwargs):
        raise ValueError('Current native path installs before selected origin; no legacy end handoff')
    def remember_replay_boundary(self,*,device_uuid,model_sha256):
        return self.remember_legacy_boundary(device_uuid=device_uuid,model_sha256=model_sha256)
    def install_at_selected_query(self,*,tick,device_uuid,model_sha256):
        self._identity(device_uuid,model_sha256)
        if (self.installed or self._legacy_hash is None or self.ledger.origin_seen
                or not self.ledger.awaiting_execution
                or tick!=self.ledger.records[-1]['tick']):
            raise ValueError('Install once at the actual selected query before original execution')
        before=self.source.router.capture(self.lane)
        if torch_hash(before)!=self._legacy_hash:
            raise ValueError('Unknown Torch draws since replay boundary')
        if torch_hash(self._snapshot)!=self._snapshot_hash:
            raise ValueError('Fresh same-process sampler snapshot mutated')
        if environment_hash(before)!=environment_hash(self._snapshot):
            raise ValueError('Replay environment RNG differs from fresh discovery; do not overwrite')
        if len(self._snapshot['torch_cuda'])!=(1 if self.source.router.cuda else 0):
            raise ValueError('Exactly one selected device stream required')
        torch.set_rng_state(self._snapshot['torch_cpu'].clone())
        if self.source.router.cuda:
            torch.cuda.set_rng_state_all([x.clone() for x in self._snapshot['torch_cuda']])
        after=self.source.router.capture(self.lane)
        if environment_hash(before)!=environment_hash(after) or torch_hash(after)!=self._snapshot_hash:
            self.poisoned=True
            raise ValueError('Torch-only selected-query fork failed')
        self.installed=True
        self.handoff=dict(branch=self.name,owner_pid=self.source.owner_pid,nonce=self.source.nonce,
             mode='current_RGB_fresh_fork_before_selected_origin',
             device_uuid=device_uuid,model_sha256=model_sha256,
             query_count=len(self.ledger.records),base_sha256=self.ledger.records[-1]['base_sha256'],
             tick=tick,environment_unchanged=True,
             replay_torch_hash=torch_hash(before),fresh_sampler_torch_hash=torch_hash(after),
             old_legacy_combined_RNG_equality_not_claimed=True)
        return copy.deepcopy(self.handoff)
    def finish_origin(self,returned,*,end_tick,requested95_arrived):
        if not self.installed:
            raise ValueError('Selected origin cannot execute before fresh sampler fork')
        return super().finish_origin(returned,end_tick=end_tick,requested95_arrived=requested95_arrived)
