"""Deploy an audited S2m head through the unchanged S2j sampler/input path."""
from pathlib import Path

import numpy as np

from .audit import digest
from .structured_action_adapter import FORWARD_KEYS, StructuredActionAdapter
from .structured_action_attribution_policy import StructuredAttributionPolicy
from .train_expert_history_action import BASE_RUN, CONDITIONS, read_checkpoint


class ExpertHistoryActionPolicy(StructuredAttributionPolicy):
    """Head replacement after explicit certificate checks, before episode RNG.

    All encoding, sole base query, history, limits and numeric recording delegate
    to the frozen implementation. No runtime monkeypatch or extra query occurs.
    Public evaluation condition labels are metadata, never model inputs.
    """
    def __init__(self, base, checkpoint, condition):
        if condition not in ('L0','S0','S_expert_history','S0_zero','S_expert_history_zero'):
            raise ValueError('Unknown paired expert-history condition')
        underlying = {'S_expert_history':'S0','S_expert_history_zero':'S0_zero'}.get(condition,condition)
        super().__init__(base, BASE_RUN/'S0_seed0.pt', underlying)
        self.evaluated_condition=condition
        self.derived_checkpoint_sha256=None
        if condition in ('S_expert_history','S_expert_history_zero'):
            payload=read_checkpoint(checkpoint,CONDITIONS[1])
            meta=payload['metadata']
            for key in ('channels','geometry_mean','caps_native_units','projection_path',
                        'projection_sha256','robot_config_path','robot_config_sha256','parameters'):
                if meta[key]!=self.metadata[key]:
                    raise ValueError('Derived and frozen deployment contracts differ: '+key)
            head=StructuredActionAdapter(meta['channels'],meta['geometry_mean'])
            head.load_state_dict(payload['model'],strict=True)
            head.eval().requires_grad_(False)
            np.testing.assert_array_equal(head.caps.numpy(),self.frozen_model.caps.numpy())
            self.model.frozen_model=head
            self.derived_checkpoint_sha256=digest(Path(checkpoint))

        # Observe the actual existing forward inputs; no second encoding/query.
        self.capture_handle = self.model.register_forward_pre_hook(self._capture_legal)

    def reset_history(self):
        super().reset_history()
        self.last_legal_inputs = None

    def _capture_legal(self, module, args):
        self.last_legal_inputs = None
        if self.query_count >= 2:
            return
        inputs, base = args
        if set(inputs) != FORWARD_KEYS:
            raise ValueError('Capture requires exactly nine legal fields')
        self.last_legal_inputs = {key:value.detach().cpu().numpy().copy() for key,value in inputs.items()}
        self.last_legal_inputs['base'] = base.detach().cpu().numpy().copy()

    def predict(self,*args,**kwargs):
        output=super().predict(*args,**kwargs)
        self.last_diagnostics.update(evaluated_condition=self.evaluated_condition,
            derived_checkpoint_sha256=self.derived_checkpoint_sha256)
        return output
