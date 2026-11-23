"""One sampled base, two immutable CPU readouts, original TTA coordinates."""
import numpy as np
import torch
from cx_tta.eval_diagnostics import array_hash

class CompositeHead(torch.nn.Module):
    def __init__(self,gate,auxiliary):
        super().__init__();self.gate=gate;self.auxiliary=auxiliary
        self.last_gate=None
        self.eval().requires_grad_(False)

    @property
    def action_head(self):
        # The unchanged ActorQuery hook still records the exact qualified
        # auxiliary hidden/logits, without replacing its nested owner callables.
        return self.auxiliary.action_head

    def forward(self,legal,base,enabled=True):
        if self.last_gate is not None:raise ValueError('Unconsumed gate candidate')
        captured={}
        def hook(name):
            def record(module,args,value):captured[name]=value.detach().cpu().numpy().copy()[0]
            return record
        hooks=[self.gate.action_head[1].register_forward_hook(hook('hidden')),self.gate.action_head[2].register_forward_hook(hook('logits'))]
        try:
            gate_result=self.gate(legal,base,enabled=enabled)
            self.last_gate=dict(raw_delta=gate_result['delta'][0].detach().cpu().numpy().copy(),base=base[0].detach().cpu().numpy().copy(),**captured)
        finally:
            for h in hooks:h.remove()
        return self.auxiliary(legal,base,enabled=enabled)

    def take(self,base):
        if self.last_gate is None:raise ValueError('No current gate readout')
        value=self.last_gate;self.last_gate=None
        assert np.array_equal(base,value['base'])
        return value
