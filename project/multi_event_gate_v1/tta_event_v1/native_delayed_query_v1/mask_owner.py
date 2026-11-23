"""Single actual SAM2 owner; query boundaries reuse the same arrived end frame.

Only the earliest semantic seed may be reused by an exact RGB/language bridge.
All later masks must be recomputed from this owner's actual arrived RGB prefix.
"""
import hashlib
import numpy as np
from native_arrived_mask_v3.stream import NativeMaskStream

class ArrivedMaskOwner:
    def __init__(self, model, earliest_masks, *, core=None):
        self.seed=None if earliest_masks is None else np.asarray(earliest_masks).copy()
        if self.seed is not None and (self.seed.dtype!=bool or self.seed.shape!=(2,240,320)):
            raise ValueError('Exact earliest two-ID seed or explicit unknown required')
        self.memory=NativeMaskStream(model,core=core)if self.seed is not None else None
        self.latest=None;self.query=-1;self.origins=[];self.closed=False
        self.actual_tracking_calls=0;self.boundary_reuses=0

    def _frame(self,rgb,*,tick,elapsed_s,kind):
        if self.closed or rgb.dtype!=np.uint8 or rgb.shape!=(240,320,3):
            raise ValueError('Actual current original head RGB required')
        if type(tick)is not int or tick<0 or not np.isfinite(elapsed_s)or elapsed_s<0:
            raise ValueError('Actual nonnegative physical clock required')
        digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        if self.latest is not None:
            if tick==self.latest['global_tick']:
                if kind!='query' or digest!=self.latest['RGB_sha256']or elapsed_s!=self.latest['elapsed_episode_s']:
                    raise ValueError('Conflicting same-frame RGB/time or duplicate native step')
                self.boundary_reuses+=1
                return self.latest
            if tick<self.latest['global_tick']or elapsed_s<=self.latest['elapsed_episode_s']:
                raise ValueError('Backward/future observation ordering')
        elif tick!=0 or elapsed_s!=0 or kind!='query':
            raise ValueError('Initialize only at actual earliest query0')
        if self.memory is not None:
            if self.latest is None:
                row=self.memory.initialize(rgb,self.seed,global_tick=0,elapsed_s=0.,decision_tick=0)
            else:
                row=self.memory.append(rgb,global_tick=tick,elapsed_s=elapsed_s,decision_tick=tick,kind=kind)
            self.actual_tracking_calls+=1
        else:
            row=dict(global_tick=tick,elapsed_episode_s=float(elapsed_s),kind=kind,RGB_sha256=digest,
                     masks=None,presence=None,raw_logits_sha256=None,reason='earliest_semantic_identity_unknown')
        self.latest=row
        return row

    def at_query(self,rgb,*,query,tick,elapsed_s):
        if type(query)is not int or query!=self.query+1:
            raise ValueError('One contiguous actual query boundary')
        row=self._frame(rgb,tick=tick,elapsed_s=elapsed_s,kind='query')
        self.query=query
        return row

    def mark_origin(self):
        if self.closed or self.latest is None or self.query<0 or self.query in self.origins or len(self.origins)>=3:
            raise ValueError('One source per actual current query, max3')
        # The selected event is bookkeeping; never reset temporal object memory.
        self.origins.append(self.query)

    def after_native(self,rgb,*,tick,elapsed_s,kind):
        if not self.origins or kind not in('origin','following'):
            raise ValueError('Declared current native segment required')
        return self._frame(rgb,tick=tick,elapsed_s=elapsed_s,kind=kind)

    def finish(self):
        self.closed=True
        return dict(actual_tracking_calls=self.actual_tracking_calls,boundary_frame_reuses=self.boundary_reuses,
                    known_earliest_identity=self.seed is not None,source_events=len(self.origins))
