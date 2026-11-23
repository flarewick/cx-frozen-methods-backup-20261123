"""Private copies of audited, unused original frozen wrapper blueprints."""
import copy
import json

class FrozenFactory:
    def __init__(self,original):self.original=original;self.templates={}
    def __call__(self,base,adapter,condition,config):
        key=(adapter,condition,json.dumps(config.to_dict(),sort_keys=True))
        if key not in self.templates:
            template=self.original(base,adapter,condition,config)
            if template.previous is not None or template.query_count:raise RuntimeError('Template already used')
            self.templates[key]=template
        result=copy.deepcopy(self.templates[key],memo={id(base):base,id(base._model):base._model,id(base._sample_actions):base._sample_actions})
        if result.base_policy is not base or result.sampler is not base._sample_actions or result.previous is not None or result.query_count:raise RuntimeError('Fresh policy clone contract failed')
        return result
