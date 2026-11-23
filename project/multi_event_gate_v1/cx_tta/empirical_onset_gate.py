"""Frozen empirical-score selector; not calibrated benefit/risk bounds."""
from pathlib import Path
import numpy as np
from .audit import digest
from .native_history_cache import read,require,verify
from .validate_shared_onset_readout import load_model
from .onset_observables import validate_packet
from .structured_action_adapter import ARM_INDICES
from .arm_prescription_policy import compose

GATES={'G_candidate_cost2':'candidate_conditioned_four_class',
       'G_observation_cost2':'observation_base_four_class'}


def load_frozen_gate(run,condition):
    require(condition in GATES,'Fixed empirical gate variant')
    run=Path(run);p=read(run/'protocol.json');result=read(run/'result.json')
    require(result['status']=='complete' and result['progression']['signal_gate_passed'] and read(run/'independent_verification.json')['passed'],'Audited pilot signal')
    verify(read(run/'pre_calibration_seal.json')['model_and_OOF_sha256'])
    model=load_model(run/GATES[condition])
    require(model.metadata['fit_indices']==p['recipe']['fit_rows'] and model.metadata['calibrated'] is False,'Frozen fit-only uncalibrated model')
    return model,digest(run/GATES[condition]/'model.npz')


def candidate_packet(legal,raw):
    delta=np.zeros_like(raw)[None]
    delta[:,:,ARM_INDICES]=raw[None][:,:,ARM_INDICES]
    return validate_packet(dict(legal,candidate_delta=delta))


class EmpiricalOnsetGate:
    def __init__(self,model):self.model=model;self.reset()
    def reset(self):self.last_query=-1;self.onset=None;self.consumed=False
    def step(self,query,window,packet=None):
        require(type(query)is int and query==self.last_query+1,'Contiguous gate queries')
        if window is not None:
            require(len(window)==2 and all(type(v)is int for v in window) and window[0]>=1 and window[1]==window[0]+2,'Original opportunity window')
            if self.onset is None:self.onset=window[0]
            require(self.onset==window[0],'One latched opportunity')
        elif self.onset is not None:raise ValueError('Latched opportunity disappeared')
        due=self.onset is not None and query==self.onset and not self.consumed
        require(due or packet is None,'No readout outside the sole onset')
        scores=None;margin=None;accept=False
        if due:
            legal=validate_packet(packet);prediction=self.model.predict(legal);scores=prediction['class_scores']
            require(set(scores)=={'both_fail','rescue','harm','both_success'} and all(np.isfinite(v) and 0<=v<=1 for v in scores.values()) and abs(sum(scores.values())-1)<1e-5,'Finite four-class scores')
            margin=float(scores['rescue']-2*scores['harm']);accept=bool(margin>0);self.consumed=True
        self.last_query=query
        return dict(scored=bool(due),accepted=accept,consumed=self.consumed,onset=self.onset,
            scores=scores,margin=margin,harm_cost=2,threshold=0,calibrated=False,rule='empirical_score_only')


def selected_actions(base,raw,query,window,decision):
    require(type(decision['accepted'])is bool and (not decision['accepted'] or decision['scored']),'Accept only a scored onset')
    return compose(base,raw,'F_first_full' if decision['accepted'] else 'Z_zero',query,window)
