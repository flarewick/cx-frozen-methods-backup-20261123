"""Original new-adapter closures with independent scenes and one sampler owner."""
import copy,functools,json,threading
from pathlib import Path
from concurrent_core import Coordinator,PolicyGate,closure_set
from frozen_factory import FrozenFactory

class Shared6Coordinator(Coordinator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.parity_count={};self.parity_path=None;self.cache_export=None;self.screen_references=dict(expected={},sources=[]);self.screen_checks=[];self.screen_parity_path=None
    def screen(self,lane,original,seed,index):
        def checked(*args):
            row=original(*args)
            from shared6_screen_reference import verify
            if verify(row,self.screen_references):
                self.screen_checks.append(seed)
                if self.screen_parity_path:self.screen_parity_path.write_text(json.dumps(dict(passed=True,original_real_screen_seeds=self.screen_checks,source_references=self.screen_references['sources'],no_reference_admission_or_policy_outcomes_used=True))+'\n')
                self.note('original_expert_screen_reference_exact',seed=seed)
            return row
        return super().screen(lane,checked,seed,index)
    def prepare_task(self,task,lane):
        def wrap(name,setup=False,exclusive=False):
            original=getattr(task,name)
            @functools.wraps(original)
            def invoke(*args,**kwargs):
                self.check_cancel()
                if setup:
                    def execute():
                        with self.scene_gate.write():
                            result=original(*args,**kwargs)
                            from native_generic_physics import install_task
                            from native_render import install_task as render
                            install_task(task,self.scene_gate,self.native);render(task,self.scene_gate,self.native,self.note)
                            from shared6_render_sync import install as visual_sync
                            visual_sync(task,self.note)
                            return result
                    execute._cx_priority=4;execute._cx_stage='original_task_setup';return self.call(lane,execute)
                with(self.scene_gate.write()if exclusive else self.scene_gate.read()):return original(*args,**kwargs)
            setattr(task,name,invoke)
        wrap('setup_demo',setup=True)
        for name in('get_obs','_update_render'):wrap(name)
        wrap('close_env',exclusive=True);return task

class AuditedPolicyGate(PolicyGate):
    def predict(self,*args,**kwargs):
        c=self._coordinator;lane=self._lane;wrapped=self._wrapped
        def invoke():
            import numpy as np,torch
            condition=wrapped.condition
            if c.parity_count.get(condition,0)>=3:
                value=wrapped.predict(*args,**kwargs);c.verify_weights();return value
            before=c.rng.capture(lane);base=wrapped.base_policy
            reference=copy.deepcopy(wrapped,memo={id(base):base,id(base._model):base._model,id(base._sample_actions):base._sample_actions})
            value=wrapped.predict(*args,**kwargs);after=c.rng.capture(lane);c.rng.restore(lane,before);expected=reference.predict(*args,**kwargs);again=c.rng.capture(lane)
            from cx_tta.recovery_contract import rng_fingerprint
            if rng_fingerprint(after)!=rng_fingerprint(again)or any(not np.array_equal(a,b)for a,b in zip(value,expected)):raise RuntimeError('Original trained-policy GPU actions/delta/base/RNG parity failed')
            if set(wrapped.last_legal_inputs)!=set(reference.last_legal_inputs):raise RuntimeError('Observable feature keys differ')
            for key,actual in wrapped.last_legal_inputs.items():
                target=reference.last_legal_inputs[key]
                if torch.is_tensor(actual):same=torch.equal(actual,target)
                else:same=np.array_equal(actual,target)
                if not same:raise RuntimeError('Original trained-policy feature differs: '+key)
            c.rng.restore(lane,after);c.verify_weights();c.parity_count[condition]=c.parity_count.get(condition,0)+1
            if c.parity_path:c.parity_path.write_text(json.dumps(dict(passed=True,queries_by_condition=c.parity_count,features_actions_delta_base_rng_exact=True))+'\n')
            c.note('original_new_adapter_query_parity',condition=condition,queries=c.parity_count[condition],passed=True);return value
        invoke._cx_priority=0;invoke._cx_stage='original_trained_adapter_predict';return c.call(lane,invoke)

def configure(callbacks,c,lane):
    from cx_tta.multi_event import EventConfig
    rollout=callbacks['rollout'];c.prepare_task(callbacks['task'],lane)
    closure_set(rollout,'create_base_policy',lambda:c.get_base(lane))
    if c.wrapper_factory is None:c.wrapper_factory=FrozenFactory(callbacks['StackingPolicy'])
    def make(base,training_run,condition,config=EventConfig()):
        def create():return AuditedPolicyGate(c.wrapper_factory(base,training_run,condition,config),c,lane)
        create._cx_priority=3;create._cx_stage='private_trained_policy_wrapper';return c.call(lane,create)
    closure_set(rollout,'StackingPolicy',make)
    original_probe=callbacks['steady_probe']
    def probe(*args,**kwargs):
        def execute():
            with c.scene_gate.write():
                result=original_probe(*args,**kwargs)
                if c.cache_export:c.cache_export(result,args[2],args[3])
                return result
        execute._cx_priority=1;execute._cx_stage='original_zero_L0_steady_probe';return c.call(lane,execute)
    closure_set(rollout,'steady_probe',probe)
    screen=callbacks['screen']
    callbacks['screen']=lambda seed,index:c.screen(lane,screen,seed,index)
    return callbacks
