"""Read-only frozen phase auxiliary; no fast128 or policy update here."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
T=Path(__file__).resolve().parent.parent

class FrozenPredictor:
    def __init__(self):
        root=T/'outputs/native_phase_feedback_v1/frozen_auxiliary'
        self.protocol=json.loads((root/'protocol.json').read_text())
        file=root/'predictor.npz'
        assert hashlib.sha256(file.read_bytes()).hexdigest()==self.protocol['checkpoint_sha256']
        assert self.protocol['auxiliary_candidate_frozen']and self.protocol['primary_model']=='correct'
        assert self.protocol['ridge']==1. and not self.protocol['direction_training_allowed']
        path=T/'native_local_response_v2/model.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==self.protocol['source_sha256'][str(path)]
        spec=importlib.util.spec_from_file_location('fixed_deployed_local_response_model',path)
        self.model=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.model)
        with np.load(file,allow_pickle=False)as z:self.state={k:z[k].copy()for k in z.files}
        for value in self.state.values():value.setflags(write=False)

    def predict(self, rows):
        if not rows:return np.empty((0,2),np.float64)
        x=np.stack([r['context']for r in rows])
        u=np.stack([r['command']for r in rows])
        points=np.stack([r['start_points']for r in rows])
        if not np.isfinite(x).all()or not np.isfinite(u).all():
            raise ValueError('Finite observed context/own commands required')
        xx,v,kind,_=self.model.coordinates(x,u,points,'correct')
        return self.model.predict(self.state,xx,v,kind)
