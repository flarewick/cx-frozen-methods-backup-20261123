"""Serial GPU-owner SAM2 math profile, restored before the next VLA call.

Standalone teacher default is highest. VLA explicitly sets high in its model
constructor. Actual raw-logit bridges must still prove numeric equivalence.
"""
from contextlib import contextmanager
import torch

@contextmanager
def sam2_profile():
    previous=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:yield
    finally:torch.set_float32_matmul_precision(previous)
