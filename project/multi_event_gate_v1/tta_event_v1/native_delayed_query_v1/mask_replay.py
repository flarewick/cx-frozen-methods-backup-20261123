"""Actual52 RGB/readouts replay into owner control, not a SAM2 forward proof."""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
from native_delayed_query_v1.mask_owner import ArrivedMaskOwner
C=Path(__file__).resolve().parent;T=C.parent;O=T/'outputs/native_delayed_query_v1'
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image(rgb):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

class SavedReadouts:
    """External saved-core substitute, only for CPU owner/controller verification."""
    def __init__(self,record):self.z=load(record['masks_file']);self.record=record;self.i=-1
    def next(self,rgb,tick):
        self.i+=1;z=self.z
        assert int(z['global_ticks'][self.i])==tick
        assert image(rgb)==self.record['RGB_sha256'][self.i]
        return dict(masks=z['masks'][self.i],presence=z['presence'][self.i],
                    raw_logits_sha256=self.record['raw_logits_sha256'][self.i])
    def initialize(self,rgb,masks,*,tick,decision_tick):
        assert self.i==-1 and tick==decision_tick==0
        return self.next(rgb,tick)
    def append(self,rgb,*,tick,decision_tick):
        assert tick==decision_tick
        return self.next(rgb,tick)

def run():
    sp=importlib.util.spec_from_file_location('mask_owner_replay_actual_data',T/'native_arrived_mask_v3/data.py')
    d=importlib.util.module_from_spec(sp);sp.loader.exec_module(d)
    spec=read(T/'outputs/native_arrived_mask_v3/protocol.json')
    root=T/'outputs/native_arrived_mask_v3/collections/79fa25d4427ec9cd476539a2'
    records=read(root/'GPU_work_complete.json')['rows'];checks=0;rows=[]
    for job,record in zip(spec['jobs'],records):
        data=d.LoadedStream(job);seed=data.seed_masks()
        core=SavedReadouts(record)if seed is not None else None
        owner=ArrivedMaskOwner(None,seed,core=core)
        count=0
        for query,(rgb,meta)in enumerate(data.queries()):
            got=owner.at_query(rgb,query=query,tick=meta['global_tick'],elapsed_s=meta['elapsed_s'])
            assert got['RGB_sha256']==image(rgb);checks+=1
            if seed is not None:
                assert np.array_equal(got['masks'],core.z['masks'][count]);checks+=1
            count+=1
        owner.mark_origin();reuse=0
        for images,meta in data.native():
            got=owner.after_native(images[0],tick=meta['global_tick'],elapsed_s=meta['elapsed_s'],kind=meta['kind'])
            assert got['RGB_sha256']==image(images[0]);checks+=1
            if seed is not None:
                assert np.array_equal(got['masks'],core.z['masks'][count]);checks+=1
                assert np.array_equal(got['presence'],core.z['presence'][count]);checks+=1
            count+=1
            origin=job['stream']['segments'][0]
            if meta['global_tick']==origin['annotation_available_tick']and job['stream']['following_query_sampled']:
                before=owner.actual_tracking_calls
                at_query=owner.at_query(images[0],query=len(data.history['queries']),tick=meta['global_tick'],elapsed_s=meta['elapsed_s'])
                assert at_query is got and owner.actual_tracking_calls==before;checks+=1;reuse+=1
                # A modified same-tick frame is rejected before tracker/owner mutation.
                changed=images[0].copy();changed[0,0,0]^=1
                try:owner._frame(changed,tick=meta['global_tick'],elapsed_s=meta['elapsed_s'],kind='query')
                except ValueError:checks+=1
                else:raise AssertionError('Conflicting current boundary admitted')
        if seed is not None:
            assert core.i+1==len(core.z['masks'])==owner.actual_tracking_calls==count;checks+=1
        result=owner.finish()
        assert result['boundary_frame_reuses']==reuse;checks+=1
        rows.append(dict(stream_index=job['index'],**result))
    report=dict(passed=True,checks=checks,streams=len(rows),rows=rows,
        actual_recorded_q1_boundary_reuses=sum(r['boundary_frame_reuses']for r in rows),
        known_earliest_streams=sum(r['known_earliest_identity']for r in rows),
        source_sha256={str(p):sha(p)for p in(C/'mask_owner.py',C/'mask_replay.py',T/'native_arrived_mask_v3/stream.py')},
        input_sha256={str(p):sha(p)for p in(T/'outputs/native_arrived_mask_v3/protocol.json',root/'GPU_work_complete.json')},
        scope='Saved core substitute plays actual teacher readouts, verifying owner clock/dedup/unknown handling only. No actual SAM2 forwards. Old recorded q1 has unverified full-policy terminal permission; q2 is not called or claimed.',
        real_tracker_GPU_calls=0,actual_third_queries=0,online_updates=0,TTA_utility_proven=False)
    p=O/'mask_owner_replay.json';text=json.dumps(report,indent=2)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    print('ACTUAL52_MASK_OWNER_BOUNDARY_REPLAY_PASS',checks,report['actual_recorded_q1_boundary_reuses'],flush=True)
if __name__=='__main__':run()
