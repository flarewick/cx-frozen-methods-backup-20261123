"""Keep the earlier inference contract, changing only its semantic namespace."""
import types
from native_object_association_v1.inference import infer_or_missing as _original, rng_digest
from . import semantic

infer_or_missing = types.FunctionType(_original.__code__, dict(_original.__globals__, semantic=semantic), _original.__name__, _original.__defaults__, _original.__closure__)
