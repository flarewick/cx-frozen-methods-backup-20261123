"""Strengthen the existing three-query gate parity with original auxiliary encoding.

Actual policy values remain untouched. Only the already existing reference clone
uses the archived original eager auxiliary prefix. A mismatch saves actual input
and both legal arrays before the original failure propagates.
"""
import ast,copy,inspect,linecache,os,time,textwrap
from pathlib import Path
_LAST=None
_INSTALLED=False
RECEIPT=None

def oracle_encode(policy,images,state,prompt):
 global _LAST
 from cx_tta.cache_joint_prefix import encode
 value,meta=encode(policy,images,state,prompt)
 _LAST=dict(images=[x.copy()for x in images],state=state.copy(),prompt=prompt,joint=value.copy(),metadata=meta)
 return value,meta

def capture(c,lane,actual,reference,key):
 import json,numpy as np,torch
 import benefit_followup_protocol_v2 as p
 from benefit_followup_stream_v3 import atomic
 if _LAST is None:raise RuntimeError('Original-reference input capture missing')
 folder=p.ROOT/'diagnostics'/('original_reference_mismatch_'+str(time.time_ns()));folder.mkdir(parents=True)
 values=dict(head=_LAST['images'][0],right=_LAST['images'][1],left=_LAST['images'][2],state=_LAST['state'],original_eager_joint=_LAST['joint'])
 def array(value):return value.detach().cpu().numpy().copy()if torch.is_tensor(value)else np.asarray(value).copy()
 for field,value in actual.last_legal_inputs.items():values['actual__'+field]=array(value)
 for field,value in reference.last_legal_inputs.items():values['original__'+field]=array(value)
 np.savez(folder/'actual_inputs_and_legal_arrays.npz',**values)
 atomic(folder/'receipt.json',dict(complete=True,diagnostic_only=True,condition=actual.condition,seed=actual.episode_seed,query=actual.last_diagnostics['query'],mismatched_field=key,lane=lane.name,prompt=_LAST['prompt'],metadata=_LAST['metadata'],arrays_sha256=p.sha(folder/'actual_inputs_and_legal_arrays.npz'),helper_sha256=p.sha(__file__),actual_policy_values_never_replaced=True,original_error_must_propagate=True))
 print('ORIGINAL_REFERENCE_MISMATCH_CAPTURED',str(folder),flush=True)

def install():
 global _INSTALLED,RECEIPT
 if _INSTALLED:return RECEIPT
 import shared6_eval_core as core
 from concurrent_core import closure_set
 wrapper=core.AuditedPolicyGate.predict
 if 'predict'not in wrapper.__code__.co_freevars:raise RuntimeError('Expected archived v7 prefix activation wrapper')
 fn=wrapper.__closure__[wrapper.__code__.co_freevars.index('predict')].cell_contents
 node=ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0];before=copy.deepcopy(node);added=[]
 for parent in ast.walk(node):
  body=getattr(parent,'body',None)
  if not isinstance(body,list):continue
  for i in range(len(body)-1,-1,-1):
   statement=body[i]
   if isinstance(statement,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='reference'for t in statement.targets)and isinstance(statement.value,ast.Call)and ast.unparse(statement.value.func)=='copy.deepcopy':
    new=ast.parse("reference.encode_eager=torch.compiler.disable(__import__('benefit_prefix_original_reference_v11').oracle_encode)").body[0];body.insert(i+1,new);added.append(new)
   if isinstance(statement,ast.Raise):
    expression=ast.unparse(statement);field=None
    if 'Original trained-policy feature differs:'in expression:field='key'
    elif 'Original trained-policy GPU actions/delta/base/RNG parity failed'in expression:field=repr('GPU_actions_delta_base_or_RNG')
    elif 'Observable feature keys differ'in expression:field=repr('observable_feature_keys')
    if field is not None:
     new=ast.parse("__import__('benefit_prefix_original_reference_v11').capture(c,lane,wrapped,reference,"+field+")").body[0];body.insert(i,new);added.append(new)
 if len(added)!=4:raise RuntimeError('Original parity reference boundaries changed')
 restored=copy.deepcopy(node)
 for parent in ast.walk(restored):
  body=getattr(parent,'body',None)
  if isinstance(body,list):body[:]=[n for n in body if not(isinstance(n,(ast.Assign,ast.Expr))and 'benefit_prefix_original_reference_v11'in ast.unparse(n))]
 if ast.dump(restored,include_attributes=False)!=ast.dump(before,include_attributes=False):raise RuntimeError('Actual policy/three-query parity/error body changed')
 source=ast.unparse(ast.fix_missing_locations(node))+'\n';file='<original_reference_parity_v11>';linecache.cache[file]=(len(source),None,source.splitlines(True),file)
 namespace=dict(fn.__globals__);exec(compile(source,file,'exec'),namespace);closure_set(wrapper,'predict',namespace[fn.__name__])
 RECEIPT=dict(actual_policy_and_original_error_AST_unchanged=True,existing_three_query_limit_unchanged=True,reference_clone_only_uses_original_auxiliary_prefix=True,reference_mismatch_inputs_saved_before_original_error=True)
 _INSTALLED=True;return RECEIPT
