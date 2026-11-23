"""Frozen all-event v118 remains authoritative for each actual proposed action."""
import copy,sys
from types import SimpleNamespace
import numpy as np
from .contract import W,GATE_MODEL,read
from .paired_heads import ARM

class EventComposer:
    def __init__(self,config):
        from cx_tta.multi_event import EventDetector,LocalController
        from scripts.gate_all_candidate_v237.allgate237 import AllCandidateVeto
        self.detector=EventDetector(config);self.controller=LocalController(config)
        self.gate=AllCandidateVeto(self.controller,read(GATE_MODEL));self.query=0;self.history={};self.current=None

    def step(self,base,reference,legal,*,adapted_ARM=None):
        from cx_tta.empirical_onset_gate import candidate_packet
        from cx_tta.multi_event import apply_arm_candidate
        from candidate_eval_policy import scaled_actions
        from benefit_fixed_ensemble_gate_v119 import vector
        direction=reference.copy()
        if adapted_ARM is not None:
            if adapted_ARM.shape!=(32,12)or adapted_ARM.dtype!=np.float32 or not np.isfinite(adapted_ARM).all():raise ValueError('Finite transferred gate ARM required')
            direction[:,ARM]=adapted_ARM
        # The gate sees precisely the candidate that would be executed, rather
        # than scoring an old direction and silently applying a changed one.
        packet=candidate_packet(legal,direction)
        event,evidence=self.detector.step(self.query,packet)
        scores=lambda p,c:dict(both_fail=0.,rescue=1.,harm=0.,both_success=0.)
        decision=self.gate.step(self.query,event,packet,scores,enabled=True)
        _,raw=apply_arm_candidate(base,direction,decision['active'])
        actions,delta=scaled_actions(base,raw,.5)
        records=[dict(row,actual_candidate_vector593=vector(packet,row['context']).tolist())for row in self.gate.last_veto_records]
        meta=dict(query=self.query,event=event,evidence=evidence,decision=decision,adapted_direction_supplied=adapted_ARM is not None,adaptation_applied=bool(decision['active']and not np.array_equal(direction,reference)),all_event_gate_records=records,candidate_alpha=.5)
        self.history[self.query]=copy.deepcopy(decision['selected_event']or event)
        self.current=dict(query=self.query,legal=legal)
        self.query+=1
        assert np.array_equal(actions[:,[6,13]],base[:,[6,13]])and np.max(np.abs(delta))<=.050001
        return actions,delta,meta

class FeedbackApplication:
    """Existing fast-difference opportunity, additionally subject to frozen gate.

    Preview on CPU-local controller state so a veto cannot consume real budget
    or let an already rejected candidate slip through an application fallback.
    """
    def __init__(self):
        from native_feedback_application_v1.application import FeedbackApplication as Original
        self.state=Original()

    def step(self,composer,base,reference,direction,actions,delta,event,update,*,condition):
        from candidate_eval_policy import scaled_actions
        from cx_tta.multi_event import event_context
        from cx_tta.empirical_onset_gate import candidate_packet
        from benefit_fixed_ensemble_gate_v119 import predict,vector
        trial=copy.deepcopy(self.state);controller=copy.deepcopy(composer.controller)
        output=trial.step(SimpleNamespace(controller=controller),base,reference,direction,actions,delta,event,update,condition=condition)
        fresh,applied,metadata,receipt=output
        extra=bool(receipt.get('applied')and receipt.get('reason')=='accepted_feedback_fast_difference')
        if extra:
            source=composer.history.get(receipt['source_query']);decision=None
            if source is not None:
                assert composer.current['query']==event['query']
                context=event_context(source,event['query'],composer.controller.interventions,composer.controller.last_intervention)
                packet=candidate_packet(composer.current['legal'],applied)
                scores=predict(composer.gate.model,packet,context)
                decision=dict(query=event['query'],source_event=copy.deepcopy(source),context=context.tolist(),scores=scores,admit=bool(scores['rescue']-2*scores['harm']>0),actual_candidate_vector593=vector(packet,context).tolist(),scope='fresh_accepted_feedback_fast_difference')
            if decision is None or not decision['admit']:
                receipt=dict(receipt,applied=False,reason='missing_causal_gate_context'if decision is None else'frozen_v118_rejected_fast_difference',gate_decision=decision)
                trial.records[-1]=copy.deepcopy(receipt)
                self.state=trial
                # Real controller and its budget are unchanged by the preview.
                return actions,delta,event,receipt
            fresh,applied=scaled_actions(base,applied,.5)
            receipt=dict(receipt,gate_decision=decision,candidate_alpha=.5)
            metadata=copy.deepcopy(metadata);metadata['feedback_fast_difference']=copy.deepcopy(receipt)
        self.state=trial
        composer.controller.__dict__.clear();composer.controller.__dict__.update(controller.__dict__)
        assert composer.gate.original is composer.controller
        assert np.max(np.abs(applied))<=.050001 and np.array_equal(fresh[:,[6,13]],base[:,[6,13]])
        return fresh,applied,metadata,receipt

    def finish(self):
        return dict(self.state.finish(),all_feedback_extra_candidates_subject_to_fixed_v118=True)
