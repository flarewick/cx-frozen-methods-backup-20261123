"""Exact stack-name predicate without frame formatting; keep all type checks."""
import sys
SKIP=frozenset(('jax._src.tree_util','flax.nnx.transforms.compilation'))
_original=None

def check(self,typechecker):
 frame=sys._getframe()
 try:
  while frame is not None:
   if frame.f_globals.get('__name__')in SKIP:return None
   frame=frame.f_back
 finally:del frame
 return _original(self,typechecker)

def install():
 global _original
 import torch
 import openpi.shared.array_typing as at
 import jaxtyping._decorator as decorator
 if getattr(decorator._check_dataclass_annotations,'_benefit_fast_stack',False):return
 _original=at._original_check_dataclass_annotations
 # Validation is Python only; execute it, rather than symbolically tracing it.
 wrapped=torch.compiler.disable(check);wrapped._benefit_fast_stack=True
 decorator._check_dataclass_annotations=wrapped
 return wrapped
