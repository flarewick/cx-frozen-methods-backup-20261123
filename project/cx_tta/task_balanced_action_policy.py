"""Deploy an audited S2l head through the unchanged S2j sampler/input path."""
from pathlib import Path

import numpy as np

from .audit import digest
from .structured_action_adapter import StructuredActionAdapter
from .structured_action_attribution_policy import StructuredAttributionPolicy
from .train_task_balanced_action import BASE_RUN, CONDITIONS, read_checkpoint


class TaskBalancedActionPolicy(StructuredAttributionPolicy):
    """Head replacement after explicit certificate checks, before episode RNG.

    All encoding, sole base query, history, limits and numeric recording delegate
    to the frozen implementation. No runtime monkeypatch or extra query occurs.
    Public evaluation condition labels are metadata, never model inputs.
    """
    def __init__(self, base, checkpoint, condition):
        if condition not in ('L0','S0','S_task_balanced','S0_zero','S_task_balanced_zero'):
            raise ValueError('Unknown paired task-normalization condition')
        underlying = {'S_task_balanced':'S0','S_task_balanced_zero':'S0_zero'}.get(condition,condition)
        super().__init__(base, BASE_RUN/'S0_seed0.pt', underlying)
        self.evaluated_condition=condition
        self.derived_checkpoint_sha256=None
        if condition in ('S_task_balanced','S_task_balanced_zero'):
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

    def predict(self,*args,**kwargs):
        output=super().predict(*args,**kwargs)
        self.last_diagnostics.update(evaluated_condition=self.evaluated_condition,
            derived_checkpoint_sha256=self.derived_checkpoint_sha256)
        return output
