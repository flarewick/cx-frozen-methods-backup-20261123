"""Reuse the original one-real-object adjudicator without changing model math."""
import types
from native_object_association_v1.semantic import FrozenInstances as Previous
from native_object_feedback_v4.semantic import FrozenInstances as Descriptor
from .language import prompt, new_prompt, VERSION

_fn = Descriptor._infer_new
_infer = types.FunctionType(_fn.__code__, dict(_fn.__globals__, prompt=new_prompt, VERSION=VERSION), _fn.__name__, _fn.__defaults__, _fn.__closure__)

class FrozenInstances(Previous):
    def infer(self, images, languages):
        if len(images) != len(languages) or not images:
            raise ValueError('Matched actual RGB and instructions required')
        values = []
        for image, language in zip(images, languages, strict=True):
            if new_prompt(language)[1]:
                value = _infer(self, image, language)
                value['interface_version'] = VERSION
                value['earliest_query_identity_only'] = True
                values.append(value)
            else:
                if not prompt(language)[1]:
                    raise ValueError('Unsupported language must remain explicit missing')
                values.extend(super().infer([image], [language]))
        return values
