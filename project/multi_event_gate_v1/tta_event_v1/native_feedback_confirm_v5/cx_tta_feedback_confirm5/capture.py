"""One native32 chunk recorder. No extra action, hold, GT input or future frame."""
import numpy as np
from .paths import write,digest
from .contract import validate_packet
from .views import VIEWS
from .throughput import discard_observation
from .schedule import ACTIVE as SCHEDULE
from cx_tta_clock.clock import ACTIVE,ClockComplete
from actuation_telemetry import ACTIVE as TRACE
from native_stage_v1.clock import StageClock,OriginRecorder,COMMON_TICKS
from stage_feedback_bridge_v1.bridge import ACTIVE as FEEDBACK
from stage_feedback_v3.completion import CompletionQueue

class CurrentReadouts:
    def __init__(self,recorder,reader,*,kind=None,initial=None,base=None,on_arrival=None):
        self.recorder,self.reader=recorder,reader
        self.on_arrival=on_arrival
        self.readouts=[];self.failed=False;self.completion=None
        if kind is not None:
            self.completion=CompletionQueue()
            self.completion.register('event','origin',kind,initial['command_state'][[6,13]],base,
                 decision_tick=0,frame_id=self.frame(initial))
    @staticmethod
    def frame(obs):
        from cx_tta.eval_diagnostics import json_hash
        return json_hash(obs['images'])
    def call(self,delegate,*args,**kwargs):
        if self.failed:raise ValueError('Failed recorder cannot resume')
        before=len(self.recorder.rows)
        try:result=delegate(*args,**kwargs)
        except ClockComplete:
            self.after(before);raise
        except BaseException:self.failed=True;raise
        self.after(before);return result
    def after(self,before):
        if self.recorder.failed or len(self.recorder.rows)!=before+1:
            self.failed=True;raise ValueError('Exactly one committed native step required')
        tick=len(self.recorder.rows);dt=self.recorder.dt[-1];new95=False
        if self.completion is not None:
            was=[r['requested95_tick'] for r in self.completion.pending.values()]
            self.completion.after_actuation('origin',self.recorder.rows[-1]['gripper_logical_command'],
                  committed_tick=tick,dt_s=dt)
            now=[r['requested95_tick'] for r in self.completion.pending.values()]
            new95=was!=now
        if tick in COMMON_TICKS or new95:self.read(tick,common=tick in COMMON_TICKS,requested95=new95)
    def read(self,tick,*,common=False,requested95=False):
        if tick!=len(self.recorder.rows) or tick<=0:raise ValueError('Actual current end frame only')
        if self.readouts and self.readouts[-1]['tick']==tick:
            return self.readouts[-1]['observation']
        obs=self.reader(tick,float(sum(self.recorder.dt)))
        self.readouts.append(dict(tick=tick,elapsed_s=float(sum(self.recorder.dt)),
             common_time_readout=common,requested95_readout=requested95,observation=obs))
        if self.on_arrival is not None:
            self.on_arrival(tick,float(sum(self.recorder.dt)),obs,self.recorder)
        return obs

def capture_chunk(task,observer,initial,actions,*,max_ticks,check_cancel,kind=None,once_base=None,on_arrival=None):
    """kind only for selected origin; following is native BASE with no new event."""
    if np.asarray(actions).dtype!=np.float32 or actions.shape!=(32,14) or not np.isfinite(actions).all():
        raise ValueError('Original finite float32 native32 action schedule required')
    from cx_tta import recovery_contract as recovery
    from cx_tta.eval_diagnostics import json_hash
    clock=StageClock(max_ticks=max_ticks);recorder=OriginRecorder(max_ticks=max_ticks)
    def read(tick,elapsed):
        obs=observer.read()
        obs.update(tick=tick,elapsed_s=elapsed,physics_sha256=observer.physical(),
             rng_sha256=recovery.rng_fingerprint(recovery.rng_capture()),frame_id=None)
        obs['frame_id']=json_hash(obs['images']);return obs
    hook=CurrentReadouts(recorder,read,kind=kind,initial=initial,base=once_base,on_arrival=on_arrival)
    clock_token=ACTIVE.set(clock);trace_token=TRACE.set(recorder);schedule_token=SCHEDULE.set(None);feedback_token=FEEDBACK.set(hook)
    returned=0;censored=False;horizon=False;start_count=int(task.take_action_cnt)
    try:
        for command in actions:
            check_cancel();before=int(task.take_action_cnt)
            task.take_action(command);discard_observation(task)
            if int(task.take_action_cnt)!=before+1:censored=True;break
            returned+=1
    except ClockComplete:horizon=True
    finally:
        FEEDBACK.reset(feedback_token);SCHEDULE.reset(schedule_token);TRACE.reset(trace_token);ACTIVE.reset(clock_token)
    if not recorder.rows:
        return dict(available=False,reason='zero_native_steps',commands_returned=returned)
    terminal=hook.read(len(clock.dt))
    reason='measurement_horizon_end' if horizon else 'native_command_censored' if censored else 'origin_chunk_completed'
    recorder.seal(reason,commands_returned=returned)
    ready=[];completion=[]
    if hook.completion is not None:
        hook.completion.finish_origin('origin',arrived_tick=len(clock.dt),commands_returned=returned,
              native_censored=bool(censored or horizon))
        ready=hook.completion.observe_rgb('origin',observation_tick=len(clock.dt),decision_tick=len(clock.dt),frame_id=hook.frame(terminal))
        completion=hook.completion.closed
    samples=hook.readouts;ticks=np.array([s['tick'] for s in samples],np.int64)
    packet=dict(ticks=ticks,elapsed_s=np.array([s['elapsed_s'] for s in samples],np.float64),
        joint=np.stack([s['observation']['joint'] for s in samples]),
        visual=np.stack([s['observation']['visual'] for s in samples]),
        command_state=np.stack([s['observation']['command_state'] for s in samples]),
        commands=np.asarray(clock.commands,np.float32),dt_s=np.asarray(clock.dt,np.float64))
    validate_packet(packet)
    rgb={**{'initial_'+str(i):initial['rgb'][i] for i in range(3)},
         **{'samples_'+str(i):np.stack([s['observation']['rgb'][i] for s in samples]) for i in range(3)}}
    return dict(available=True,packet=packet,actuation=recorder.packet(),rgb=rgb,
       initial=initial,terminal=terminal,readouts=samples,origin=recorder.sealed,
       completion=completion,feedback_ready=bool(ready),commands_returned=returned,
       native_actions_started=int(task.take_action_cnt)-start_count,
       physical_contact_certified=False)

def export_chunk(folder,name,result,*,planned_actions,prefix_meta,absolute_start_tick):
    """Separate source_readout indices per segment; no splicing into legacy files."""
    from cx_tta.eval_diagnostics import array_hash
    if not result['available']:
        write(folder/(name+'_unavailable.json'),{k:v for k,v in result.items() if k!='available'});return
    packet=result['packet'];trace=result['actuation'];initial=result['initial']
    for suffix,value in (('',packet),('_actuation',trace),('_rgb',result['rgb'])):
        with (folder/(name+suffix+'.npz')).open('xb') as f:np.savez_compressed(f,**value)
    samples=[s['observation'] for s in result['readouts']]
    metadata=dict(absolute_start_tick=absolute_start_tick,local_ticks_not_global=True,
       planned_chunk_sha256=array_hash(planned_actions),file_sha256=digest(folder/(name+'.npz')),
       actuation_sha256=digest(folder/(name+'_actuation.npz')),view_order=list(VIEWS),
       initial={k:array_hash(initial[k]) for k in ('joint','visual','command_state')},
       initial_images=initial['images'],initial_tokens=initial['token_sha256'],
       samples=[{k:s[k] for k in ('tick','elapsed_s','physics_sha256','rng_sha256','view_images','token_sha256')} for s in samples],
       native_actions_started=result['native_actions_started'],complete_episode=False,outcomes_used=False,**prefix_meta)
    write(folder/(name+'.json'),metadata)
    write(folder/(name+'_receipt.json'),dict(origin=result['origin'],completion=result['completion'],
       feedback_ready=result['feedback_ready'],commands_returned=result['commands_returned'],
       physical_contact_certified=False,absolute_start_tick=absolute_start_tick,
       readouts=[{k:v for k,v in s.items() if k!='observation'} for s in result['readouts']]))
