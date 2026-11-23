"""Expand omitted exact nominal modifiers; keep earliest-frame identity tracking."""
import types
from native_object_feedback_v5.semantic import FrozenInstances as Previous
from native_object_feedback_v4.semantic import FrozenInstances as Descriptor
from native_object_feedback_v4.language import new_prompt as previous_descriptor
from native_transfer_semantics_v2.semantic import prompt as legacy
from .language import prompt as expanded,VERSION
def prompt(language):
 roles,phrases=legacy(language)
 return (roles,phrases)if phrases else expanded(language)
_fn=Descriptor._infer_new
_describe=types.FunctionType(_fn.__code__,dict(_fn.__globals__,prompt=expanded,VERSION=VERSION),_fn.__name__,_fn.__defaults__,_fn.__closure__)
class FrozenInstances(Previous):
 def infer(self,images,languages):
  values=super().infer(images,languages)
  for i,(image,language)in enumerate(zip(images,languages,strict=True)):
   if legacy(language)[1]:continue
   roles,phrases=expanded(language)
   changed=bool(phrases)and phrases!=previous_descriptor(language)[1]
   if changed:
    value=_describe(self,image,language)
    if value['available']:
     value['grounding_decision_path']='exact_previously_omitted_nominal_modifiers'
     value['prior_grounding_available']=values[i]['available'];value['prior_grounding_reason']=values[i]['reason'];values[i]=value
    else:values[i]['expanded_nominal_missing_reason']=value['reason']
   values[i]['interface_version']=VERSION;values[i]['roles']=dict(values[i]['roles'],interface_version=VERSION)
   values[i]['earliest_query_identity_only']=True
  return values
