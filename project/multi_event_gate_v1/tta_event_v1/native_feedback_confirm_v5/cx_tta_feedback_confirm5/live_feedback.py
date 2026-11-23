"""Real arrived RGB -> SAM2 -> local response -> once-only shadow feedback.

No physics, task ID, seed or outcome enters the response/queue inputs. The exact
earliest RGB/language seed is the sole reused mask; all later masks are computed.
"""
import hashlib
import time
import numpy as np
from native_delayed_query_v1.mask_owner import ArrivedMaskOwner
from native_delayed_query_v1.local_feedback import LocalFeedbackStream
from native_delayed_query_v1.boundary import ShadowBoundary,native_continue
from native_phase_feedback_v1.feedback_queue import FeedbackQueue
from .paths import write
from .sam2_core import OffloadedCore
from .model_math import sam2_profile

class LiveFeedback:
    def __init__(self,c,lane,model,seed,seed_masks,history,ticks,instruction,geometry,predictor,cache):
        self.c,self.lane,self.predictor=c,lane,predictor
        self.cpu_fixture=bool(getattr(c,'cpu_only_fixture',False))
        self.cache=cache
        earliest=history[0]['rgb']
        digest=hashlib.sha256(b''.join(np.ascontiguousarray(im).tobytes()for im in earliest)).hexdigest()
        assert digest==seed['earliest_three_camera_bundle_sha256']
        assert hashlib.sha256(instruction.encode()).hexdigest()==seed['instruction_sha256']
        assert bool(seed_masks is not None)==bool(seed['available'])
        self.owner=ArrivedMaskOwner(model,seed_masks,core=OffloadedCore(model)if seed_masks is not None else None)
        self.origin_tick=ticks[-1];self.dt=float(np.float32(.004));self.initial_episode_s=self.origin_tick*self.dt
        self.rows=[];self.missing=[];self.timing=[];self.query_records=[];self.segment=None
        for query,(row,tick)in enumerate(zip(history,ticks)):
            self.model_call(self.owner.at_query,row['rgb'][0],query=query,tick=tick,elapsed_s=tick*self.dt)
        self.owner.mark_origin()
        self.queue=FeedbackQueue();self.queue.arrive(self.origin_tick,0.,self.owner.latest['RGB_sha256'])
        self.ticket=self.queue.register(len(history)-1)
        self.local=None
        if seed_masks is not None:
            cfg,camera,robots=geometry
            self.local=LocalFeedbackStream(earliest[0],seed_masks,instruction,robots=robots,cfg=cfg,camera=camera)
            last=history[-1]
            self.local.start(last['rgb'][0],self.owner.latest['masks'],self.owner.latest['presence'],
                             last['command_state'],tick=self.origin_tick,elapsed_s=0.)
        self.previous_tick=self.origin_tick;self.last_s=0.;self.shadow=None

    def model_call(self,method,*args,**kwargs):
        def actual():
            before=self.owner.actual_tracking_calls;epoch=self.cache.epoch
            with sam2_profile():value=method(*args,**kwargs)
            if self.owner.actual_tracking_calls>before:self.cache.record('tracker',epoch)
            return value
        return self.c.call(self.lane,actual)

    def bind_fork(self,fork):
        assert self.shadow is None
        self.shadow=ShadowBoundary(fork,self.queue)

    def begin_segment(self,kind,start_tick,query):
        assert kind in('origin','following')and start_tick==self.previous_tick
        self.segment=dict(kind=kind,start_tick=start_tick,query=query)

    def arrived(self,tick,elapsed_s,observed,recorder):
        """Called once by the existing recorder, after a committed physical step."""
        s=self.segment;global_tick=s['start_tick']+tick
        start_s=self.last_s
        selected_s=self.segment_start_s+elapsed_s
        began=time.monotonic()
        current=self.model_call(self.owner.after_native,observed['rgb'][0],
             tick=global_tick,elapsed_s=self.initial_episode_s+selected_s,kind=s['kind'])
        tracker_s=time.monotonic()-began
        self.queue.arrive(global_tick,selected_s,current['RGB_sha256'])
        local_s=0.
        if self.local is not None:
            first=self.previous_tick-s['start_tick'];last=tick
            keys=('arm_position_target','arm_velocity_target','gripper_logical_command')
            window={k:np.stack([r[k]for r in recorder.rows[first:last]])for k in keys}
            window['dt_s']=np.asarray(recorder.dt[first:last],np.float64)
            began=time.monotonic()
            rows,missing=self.local.arrive(observed['rgb'][0],current['masks'],current['presence'],window,
                observed['command_state'],tick=global_tick,elapsed_s=selected_s,
                segment=s['kind'],segment_start_tick=s['start_tick'])
            local_s=time.monotonic()-began
            for row in rows:
                index=len(self.rows)
                row['command_query']=s['query'];self.rows.append(row)
                self.queue.window(self.ticket,row=index,start_tick=row['start_tick'],end_tick=row['end_tick'],
                    kind=s['kind'],command_start_tick=s['start_tick'],command_end_tick=global_tick)
            self.missing.extend(missing)
        self.timing.append(dict(global_tick=global_tick,tracker_seconds=tracker_s,local_seconds=local_s))
        self.previous_tick=global_tick;self.last_s=selected_s

    def capture_callback(self,kind,start_tick,query):
        self.begin_segment(kind,start_tick,query);self.segment_start_s=self.last_s
        return self.arrived

    def close_origin(self,result):
        closed=result['completion'];phase=closed[0].get('requested95_tick')if closed else None
        phase_s=None if phase is None else float(np.sum(result['packet']['dt_s'][:phase]))
        self.queue.close_origin(self.ticket,commands_returned=result['commands_returned'],
            complete_native32=result['origin']['complete_origin'],receipt_tick=self.previous_tick,
            phase95_tick=None if phase is None else self.origin_tick+phase,phase95_elapsed_s=phase_s)

    def draw(self,task,current,operation,*,query,delayed,device_uuid,model_sha256):
        if not native_continue(task):
            self.shadow.close();return None
        tick=self.previous_tick
        before=self.owner.actual_tracking_calls
        self.model_call(self.owner.at_query,current['rgb'][0],query=query,tick=tick,
                    elapsed_s=self.initial_episode_s+self.last_s)
        assert self.owner.actual_tracking_calls==before
        eligible=self.queue.eligible(self.ticket)
        began=time.monotonic();prediction=self.predictor.predict([self.rows[i]for i in eligible])
        predict_s=time.monotonic()-began
        value=self.c.call(self.lane,self.shadow.draw,task,current,operation,query=query,tick=tick,
                         delayed=delayed,device_uuid=device_uuid,model_sha256=model_sha256)
        assert value is not None
        packet=value[2]
        assert (packet.rows if packet is not None else())==eligible
        self.query_records.append(dict(query=query,tick=tick,rows=list(eligible),prediction=prediction.tolist(),
             prediction_seconds=predict_s,feedback_computed_before_actual_draw=True,optimizer_updates=0))
        return value[0],value[1]

    def reference_boundary(self,task,current,*,query):
        """Same arrived measurements before an independent ordinary policy draw."""
        from native_delayed_query_v1.boundary import boundary_proof
        if not native_continue(task):raise ValueError('No sequential query after native termination')
        tick=self.previous_tick;before=self.owner.actual_tracking_calls
        self.model_call(self.owner.at_query,current['rgb'][0],query=query,tick=tick,
                        elapsed_s=self.initial_episode_s+self.last_s)
        assert self.owner.actual_tracking_calls==before
        eligible=self.queue.eligible(self.ticket)
        began=time.monotonic();prediction=self.predictor.predict([self.rows[i]for i in eligible])
        seconds=time.monotonic()-began
        _,digest=boundary_proof(current,query=query,tick=tick)
        packet=self.queue.at_query(query=query,tick=tick,query_proof_sha256=digest)
        assert(packet.rows if packet is not None else())==eligible
        self.query_records.append(dict(query=query,tick=tick,rows=list(eligible),prediction=prediction.tolist(),
             prediction_seconds=seconds,feedback_computed_before_actual_draw=True,optimizer_updates=0))

    def export(self,folder,branch):
        final=self.shadow.close()if self.shadow is not None else self.queue.finish()
        if self.local is not None:self.local.finish()
        state=self.owner.finish()
        arrays=('context','command','target','start_points','end_points','point_alive','relative_target','relative_available')
        if self.rows:
            with(folder/(branch+'_live_local.npz')).open('xb')as f:
                np.savez_compressed(f,**{k:np.stack([r[k]for r in self.rows])for k in arrays})
        memory=self.owner.memory
        if memory is not None:
            with(folder/(branch+'_live_masks.npz')).open('xb')as f:
                np.savez_compressed(f,masks=np.stack([r['masks']for r in memory.rows]),
                    presence=np.stack([r['presence']for r in memory.rows]),
                    global_ticks=np.array([r['global_tick']for r in memory.rows],np.int64))
        write(folder/(branch+'_live_feedback.json'),dict(owner=state,queue=final,rows=[{k:r[k]for k in(
            'object_index','start_tick','end_tick','segment','dt_s','initial_points','matched_points','command_query')}for r in self.rows],
            missing=self.missing,timing=self.timing,queries=self.query_records,
            sampler_boundaries=self.shadow.records if self.shadow is not None else[],
            mask_readouts=[{k:r[k]for k in('global_tick','elapsed_episode_s','kind','RGB_sha256','raw_logits_sha256')}for r in memory.rows]if memory is not None else[],
            real_SAM2_model=not self.cpu_fixture,CPU_fixture_only=self.cpu_fixture,old_later_masks_imported=False,online_updates=0,TTA_utility_proven=False))
