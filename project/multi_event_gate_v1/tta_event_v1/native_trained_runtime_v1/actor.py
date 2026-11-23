"""Native environment physics/RNG are diagnostics, excluded from actor features."""
from native_policy_execution_v1.actor import ActorQuery
from native_policy_execution_v1.common import write
class AuditedActor(ActorQuery):
 def __init__(self,*args,native_diagnostic,**kwargs):
  super().__init__(*args,**kwargs);self.native_diagnostic=native_diagnostic
 def query(self,observed,record,folder):
  before=self.native_diagnostic()
  if before['physical_sha256']!=observed['physical_sha256']or before['rng_sha256']!=observed['rng_sha256']:raise ValueError('Actual current observation no longer matches native environment')
  result,row,values=super().query(observed,record,folder)
  after=self.native_diagnostic()
  if after!=before:raise ValueError('Actor encoding/sampling altered environment physics or RNG')
  proof=dict(query=record['query'],tick=record['global_tick'],**before,native_environment_unchanged=True,diagnostic_only=True,no_diagnostic_forward_inputs=True,actor_once_base_noise_witness=row['noise_witness'])
  write(folder/('query%03d_environment_audit.json'%record['query']),proof)
  row=dict(row,environment_audit=proof)
  return result,row,values
