"""Fixed gate with explicit context from actual initial own-ARM feedback."""
import copy
from types import SimpleNamespace
import numpy as np

class FeedbackApplication:
    """Existing fast-difference opportunity, additionally subject to frozen gate.

    Preview on CPU-local controller state so a veto cannot consume real budget
    or let an already rejected candidate slip through an application fallback.
    """
    def __init__(self):
        from native_feedback_application_v1.application import FeedbackApplication as Original
        self.state=Original()

    def step(self,composer,base,reference,direction,actions,delta,event,update,*,condition,source_kinds):
        from candidate_eval_policy import scaled_actions
        from cx_tta.multi_event import event_context
        from cx_tta.empirical_onset_gate import candidate_packet
        from benefit_fixed_ensemble_gate_v119 import predict,vector
        trial=copy.deepcopy(self.state);controller=copy.deepcopy(composer.controller)
        output=trial.step(SimpleNamespace(controller=controller),base,reference,direction,actions,delta,event,update,condition=condition)
        fresh,applied,metadata,receipt=output
        extra=bool(receipt.get('applied')and receipt.get('reason')=='accepted_feedback_fast_difference')
        if extra:
            source_query=receipt['source_query']
            source=composer.history.get(source_query);decision=None
            origin='observed_original_event'
            if source is None:
                from native_arm_phase_feedback_v1.readouts import KIND
                if source_query==0 and source_kinds.get(source_query)==KIND:
                    # Features only. This is never inserted into EventDetector or
                    # LocalController as an invented gripper/progress event.
                    source=dict(detected_query=source_query,kinds=[])
                    origin='observed_initial_own_ARM_commands'
            if source is not None:
                assert composer.current['query']==event['query']
                context=event_context(source,event['query'],composer.controller.interventions,composer.controller.last_intervention)
                packet=candidate_packet(composer.current['legal'],applied)
                scores=predict(composer.gate.model,packet,context)
                decision=dict(query=event['query'],source_context=copy.deepcopy(source),context_origin=origin,context=context.tolist(),scores=scores,admit=bool(scores['rescue']-2*scores['harm']>0),actual_candidate_vector593=vector(packet,context).tolist(),scope='fresh_accepted_feedback_fast_difference')
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
