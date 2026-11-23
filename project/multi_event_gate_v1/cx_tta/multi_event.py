"""Causal event opportunities and bounded local admission, independent of tasks.

Command state is NOT measured joint/contact state. Image-feature quietness plus
command intent can propose a progress-mismatch opportunity, never contact truth.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from .onset_observables import validate_packet
from .structured_action_adapter import ARM_INDICES

KINDS = ('close_left', 'close_right', 'open_left', 'open_right', 'progress_mismatch')


@dataclass(frozen=True)
class EventConfig:
    grip_on: float = .10
    grip_off: float = .04
    rearm_queries: int = 2
    reversal_queries: int = 2
    reversal_hold_steps: int = 2
    cooldown_queries: int = 1
    pending_ttl: int = 2
    max_pending: int = 4
    max_interventions: int = 3
    enable_progress: bool = False
    progress_queries: int = 2
    plan_rms_on: float = .08
    command_change_max: float = .015
    visual_change_max: float = .015

    def __post_init__(self):
        if not 0 < self.grip_off < self.grip_on <= 1:
            raise ValueError('Require gripper hysteresis 0 < off < on <= 1')
        for k in ('rearm_queries','reversal_queries','reversal_hold_steps','pending_ttl','max_pending','max_interventions','progress_queries'):
            if type(getattr(self,k)) is not int or getattr(self,k)<1:raise ValueError('Positive integer '+k)
        if self.reversal_hold_steps>32:raise ValueError('Hold steps cannot exceed one action chunk')
        if type(self.cooldown_queries)is not int or self.cooldown_queries<1:
            raise ValueError('At least one unmodified base block between corrections')
        if type(self.enable_progress)is not bool:raise ValueError('Explicit progress detector flag')
        if not all(math.isfinite(v) and v>0 for v in (self.plan_rms_on,self.command_change_max,self.visual_change_max)):
            raise ValueError('Positive finite progress thresholds')

    def to_dict(self):return asdict(self)


class EventDetector:
    def __init__(self, config=EventConfig()):self.config=config;self.reset()
    def reset(self):
        self.last_query=-1;self.sequence=0
        self.arms=[dict(latched=None,quiet=0,reverse=None,reverse_count=0) for _ in range(2)]
        self.progress_latched=False;self.progress_count=0;self.progress_quiet=0

    def step(self, query, packet):
        if type(query)is not int or query!=self.last_query+1:raise ValueError('Contiguous causal queries required')
        p=validate_packet(packet);self.last_query=query;c=self.config
        state=p['state'][0].copy();state[list(ARM_INDICES)]*=np.float32(np.pi)
        base=p['base'][0];emitted=[];measures=[]
        for arm,index in enumerate((6,13)):
            # Clip measurement only; execution uses the untouched base targets.
            current=float(np.clip(state[index],0,1));goals=np.clip(base[:,index],0,1)
            close=current-goals;opening=goals-current
            candidates=[(int(np.flatnonzero(v>=c.grip_on)[0]),k) for k,v in
                        [('close',close),('open',opening)] if np.any(v>=c.grip_on)]
            direction=min(candidates)[1] if candidates else None
            hold=0;longest=0
            if direction is not None:
                for hit in (close if direction=='close' else opening)>=c.grip_on:
                    hold=hold+1 if hit else 0;longest=max(longest,hold)
            lat=self.arms[arm];reason=None
            if direction is None:
                if max(float(close.max()),float(opening.max()))<=c.grip_off:
                    lat['quiet']+=1
                    if lat['quiet']>=c.rearm_queries:lat['latched']=None
                else:lat['quiet']=0
                lat['reverse']=None;lat['reverse_count']=0
            elif lat['latched'] is None:
                lat.update(latched=direction,quiet=0,reverse=None,reverse_count=0);reason='armed_edge'
            elif direction!=lat['latched']:
                lat['quiet']=0
                lat['reverse_count']=lat['reverse_count']+1 if lat['reverse']==direction else 1
                lat['reverse']=direction
                # A one-block opening may finish before the next query. Confirm
                # persistent goals within the CURRENT plan, not future observations.
                if longest>=c.reversal_hold_steps or lat['reverse_count']>=c.reversal_queries:
                    lat.update(latched=direction,reverse=None,reverse_count=0);reason='confirmed_reversal'
            else:lat.update(quiet=0,reverse=None,reverse_count=0)
            if reason:emitted.append(direction+'_'+('left' if arm==0 else 'right'))
            measures.append(dict(close=float(close.max()),open=float(opening.max()),intent=direction,
                                 intent_hold_steps=longest,latched=lat['latched'],quiet=lat['quiet'],reason=reason))
        progress=None
        if c.enable_progress and p['has_history'][0,0]:
            before=p['past_state'][0].copy();before[list(ARM_INDICES)]*=np.float32(np.pi)
            plan=float(np.sqrt(np.mean((base[:,ARM_INDICES]-state[list(ARM_INDICES)])**2)))
            commanded=float(np.sqrt(np.mean((state[list(ARM_INDICES)]-before[list(ARM_INDICES)])**2)))
            visual=float(np.linalg.norm(p['visual']-p['past_visual'])/(np.linalg.norm(p['past_visual'])+1e-6))
            quiet=plan>=c.plan_rms_on and commanded<=c.command_change_max and visual<=c.visual_change_max
            self.progress_count=self.progress_count+1 if quiet else 0
            self.progress_quiet=0 if quiet else self.progress_quiet+1
            if self.progress_quiet>=c.rearm_queries:self.progress_latched=False
            if self.progress_count>=c.progress_queries and not self.progress_latched:
                emitted.append('progress_mismatch');self.progress_latched=True
            progress=dict(plan_rms_rad=plan,command_change_rms_rad=commanded,visual_relative_change=visual,
                          evidence='command intent + low observable change; NOT contact measurement')
        event=None
        if emitted:
            event=dict(id=self.sequence,detected_query=query,kinds=emitted)
            self.sequence+=1
        return event,dict(grippers=measures,progress=progress)


def event_context(event, query, interventions, last_intervention):
    """Legal event/history values only; no task, seed, success or privileged state."""
    return np.asarray([float(k in event['kinds']) for k in KINDS]+[
        min(query-event['detected_query'],8)/8, min(interventions,8)/8,
        1. if last_intervention is None else min(query-last_intervention,8)/8],np.float32)


def score_margin(scores):
    keys={'both_fail','rescue','harm','both_success'}
    if set(scores)!=keys or not all(math.isfinite(v) and 0<=v<=1 for v in scores.values()) or abs(sum(scores.values())-1)>1e-5:
        raise ValueError('Finite normalized four-class scores required')
    return float(scores['rescue']-2*scores['harm'])


class LocalController:
    """Score every new event; defer conflicts with fresh inputs and a finite TTL.

    One admitted action chunk, followed by mandatory base-only cooldown. Queue
    expiry is explicit; no stale observation/action candidate is ever replayed.
    """
    def __init__(self, config=EventConfig()):self.config=config;self.reset()
    def reset(self):
        self.last_query=-1;self.last_intervention=None;self.interventions=0;self.pending=[];self.seen=set()

    def step(self, query, event, packet, scorer, *, enabled):
        if type(query)is not int or query!=self.last_query+1:raise ValueError('Contiguous controller queries required')
        if type(enabled)is not bool:raise ValueError('Explicit intervention enable required')
        p=validate_packet(packet);self.last_query=query;records=[];c=self.config
        kept=[]
        for e in self.pending:
            if query-e['detected_query']>c.pending_ttl:records.append(dict(event=e,reason='expired'))
            else:kept.append(e)
        self.pending=kept
        fresh_scores={}
        def assess(e):
            ctx=event_context(e,query,self.interventions,self.last_intervention)
            scores=scorer(p,ctx) if scorer is not None else None
            margin=None if scores is None else score_margin(scores)
            return dict(event=e,context=ctx.tolist(),scores=scores,margin=margin,accepted=margin is not None and margin>0)
        if event is not None:
            if (set(event)!={'id','detected_query','kinds'} or type(event['id'])is not int or event['id']<0 or
                event['detected_query']!=query or not event['kinds'] or len(set(event['kinds']))!=len(event['kinds']) or
                not set(event['kinds'])<=set(KINDS)):raise ValueError('Causal event schema')
            if event['id'] in self.seen:raise ValueError('Duplicate event ID')
            self.seen.add(event['id']);e=dict(event,kinds=list(event['kinds']))
            opposite={'close_left':'open_left','open_left':'close_left',
                      'close_right':'open_right','open_right':'close_right'}
            reversed_kinds={opposite[k] for k in e['kinds'] if k in opposite}
            retained=[]
            for old in self.pending:
                if reversed_kinds.intersection(old['kinds']):
                    records.append(dict(event=old,reason='superseded_by_reversal'))
                else:retained.append(old)
            self.pending=retained
            result=assess(e);fresh_scores[e['id']]=result
            records.append(dict(result,reason='new_event_scored' if scorer is not None else 'new_event_observed'))
            if not result['accepted'] or not enabled:
                records.append(dict(event=e,reason='rejected_this_event' if enabled else 'shadow_no_intervention'))
            elif self.interventions>=c.max_interventions:
                records.append(dict(event=e,reason='episode_budget_exhausted'))
            elif len(self.pending)>=c.max_pending:records.append(dict(event=e,reason='pending_queue_full'))
            else:self.pending.append(e)
        active=False;selected=None
        cooling=self.last_intervention is not None and query-self.last_intervention<=c.cooldown_queries
        if self.pending and enabled and not cooling and self.interventions<c.max_interventions:
            selected=self.pending.pop(0)
            result=fresh_scores.get(selected['id']) or assess(selected)
            active=result['accepted']
            records.append(dict(result,reason='apply_one_chunk' if active else 'rejected_on_fresh_rescore'))
            if active:self.interventions+=1;self.last_intervention=query
        return dict(active=bool(active),selected_event=selected,records=records,
            pending=[dict(e) for e in self.pending],interventions=self.interventions,
            cooldown=cooling,phase='local_one_chunk' if active else 'base',
            local_duration_chunks=1,gripper_residual_enabled=False)


def apply_arm_candidate(base, raw, active):
    if type(active)is not bool:raise ValueError('Explicit Boolean activation')
    for a in (base,raw):
        if not isinstance(a,np.ndarray) or a.shape!=(32,14) or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('Finite native float32 action chunk required')
    if np.abs(raw[:,ARM_INDICES]).max()>.100001:raise ValueError('Original arm residual cap exceeded')
    delta=np.zeros_like(base)
    if active:delta[:,ARM_INDICES]=raw[:,ARM_INDICES]
    actions=base.copy();mask=delta!=0;actions[mask]=base[mask]+delta[mask]
    return actions,delta
