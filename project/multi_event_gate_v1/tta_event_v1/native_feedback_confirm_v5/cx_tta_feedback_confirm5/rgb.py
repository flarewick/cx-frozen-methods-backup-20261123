"""Exact legacy image-copy helper; no new render, physics or encoder call."""
import numpy as np

def capture_views(views):
    if len(views) != 3:
        raise ValueError('Original3crossed views required')
    actual = views[0][0]
    if len(actual) != 3 or any((v.dtype != np.uint8 or v.ndim != 3 or v.shape[-1] != 3 for v in actual)):
        raise ValueError('Original3uint8RGB inputs required')
    return tuple((v.copy() for v in actual))
