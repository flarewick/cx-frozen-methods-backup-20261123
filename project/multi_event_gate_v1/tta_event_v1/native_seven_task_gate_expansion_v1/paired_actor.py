"""Keep original actor CodeTypes, once-base/noise, auxiliary qualification witness."""
import numpy as np
from native_trained_runtime_v1.actor import AuditedActor as Original
from native_policy_execution_v1.common import write,sha
from .composite_head import CompositeHead

class AuditedActor(Original):
    def query(self,observed,record,folder):
        assert type(self.head)is CompositeHead
        proposal,row,values=super().query(observed,record,folder)
        gate=self.head.take(proposal['base']);q=record['query']
        file=folder/('gate_policy%03d.npz'%q)
        with file.open('xb')as f:np.savez_compressed(f,**gate)
        receipt=dict(query=q,tick=record['global_tick'],gate_readout_file=str(file),gate_readout_sha256=sha(file),same_once_base_policy_file=row['policy_file'],same_once_base_policy_sha256=row['policy_sha256'],additional_VLA_or_noise_calls=0,auxiliary_hidden_logits_and_raw_remain_original_qualified_coordinates=True)
        write(folder/('gate_policy%03d.json'%q),receipt)
        row=dict(row,gate_policy=receipt)
        return dict(proposal,raw_delta=gate['raw_delta']),row,values
