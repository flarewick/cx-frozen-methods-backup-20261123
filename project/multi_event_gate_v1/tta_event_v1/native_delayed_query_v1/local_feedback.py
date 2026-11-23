"""Arrived-only local response construction. No saved targets or future masks read."""
import numpy as np
from native_delayed_query_v1.primitives import old

def partner_features(p, good, colors, obj):
    out=np.zeros(9);partner=1-obj
    if not old.supported(p,good,partner):return out
    own=p[obj,good[obj]];other=p[partner,good[partner]]
    if len(own)<3:raise ValueError('Observed own object required')
    center=own.mean(0);r=max(float(np.sqrt(np.mean(np.sum((own-center)**2,axis=1)))),1.)
    cc=other.mean(0);rr=max(float(np.sqrt(np.mean(np.sum((other-cc)**2,axis=1)))),1.)
    out[:]=np.r_[cc/320,(cc-center)/r,rr/r,colors[partner,good[partner]].mean(0),1.]
    if not np.isfinite(out).all():raise ValueError('Bad current partner features')
    return out

class LocalFeedbackStream:
    """One episode/branch owner. Only preceding image and observed labels survive.

    The caller supplies each actual contiguous command prefix once, ending at the
    current arrived RGB. Segments cannot borrow each other's commands. Masks must
    be generated from the actual arrived prefix by the external tracker.
    """
    def __init__(self, earliest_rgb, earliest_masks, instruction, *, robots, cfg, camera):
        self.prototype,self.known,_=old.surface.prototype(earliest_rgb,earliest_masks)
        self.instruction=instruction;self.robots=robots;self.cfg=cfg;self.camera=camera
        self.previous=None;self.prior_labels={};self.rows=0;self.missing=0;self.closed=False

    def frame(self, rgb, masks, presence, *, tick, elapsed_s):
        if (type(tick)is not int or tick<0 or not np.isfinite(elapsed_s)
                or rgb.dtype!=np.uint8 or rgb.shape!=(240,320,3)):
            raise ValueError('Actual physical RGB/tick required')
        if np.asarray(masks).dtype!=bool or np.shape(masks)!=(2,240,320) or np.shape(presence)!=(2,):
            raise ValueError('Current two-ID tracker masks/presence required')
        refined,valid,_=old.surface.refine(rgb,masks,presence,self.prototype,self.known,self.instruction)
        p,good,colors=old.flow.corners(rgb,refined,valid)
        return dict(rgb=rgb.copy(),mask=refined,valid=valid,p=p,good=good,colors=colors,
                    tick=tick,elapsed_s=float(elapsed_s))

    def start(self, rgb, masks, presence, command_state, *, tick, elapsed_s):
        if self.closed or self.previous is not None:raise ValueError('Start once per actual branch')
        q=np.asarray(command_state,float)
        if q.shape!=(14,)or not np.isfinite(q).all():raise ValueError('Exact current own commands required')
        self.command=q.copy()
        self.previous=self.frame(rgb,masks,presence,tick=tick,elapsed_s=elapsed_s)

    def arrive(self, rgb, masks, presence, trace_window, command_state, *,
               tick, elapsed_s, segment, segment_start_tick):
        if self.closed or self.previous is None:raise ValueError('Actual stream not running')
        if segment not in('origin','following'):raise ValueError('Known actual command segment only')
        previous=self.previous
        if (type(segment_start_tick)is not int or not segment_start_tick<=previous['tick']<tick
                or elapsed_s<=previous['elapsed_s']):
            raise ValueError('Cross-segment or backward RGB/command interval')
        now=self.frame(rgb,masks,presence,tick=tick,elapsed_s=elapsed_s)
        n=tick-previous['tick'];dt=elapsed_s-previous['elapsed_s']
        window={k:np.asarray(trace_window[k]).copy()for k in(
            'arm_position_target','arm_velocity_target','gripper_logical_command','dt_s')}
        if (window['arm_position_target'].shape!=(n,12)
                or window['arm_velocity_target'].shape!=(n,12)
                or window['gripper_logical_command'].shape!=(n,2)
                or window['dt_s'].shape!=(n,)
                or any(not np.isfinite(v).all()for v in window.values())
                or np.any(window['dt_s']<=0)or abs(float(window['dt_s'].sum())-dt)>1e-9):
            raise ValueError('Only the exact arrived own-command prefix')
        start_command=self.command.copy()
        next_command=start_command.copy()
        next_command[old.ARM]=window['arm_position_target'][-1]
        next_command[old.GRIP]=window['gripper_logical_command'][-1]
        if not np.array_equal(next_command.astype(np.float32),np.asarray(command_state,np.float32)):
            raise ValueError('Current own-command state and committed prefix disagree')
        rows=[];missing=[]
        if 25<=n<=125:
            images=np.stack([previous['rgb'],now['rgb']])
            refined=np.stack([previous['mask'],now['mask']])
            valid=np.stack([previous['valid'],now['valid']])
            p,good,colors=previous['p'],previous['good'],previous['colors']
            end,alive,_=old.flow.track(images,refined,valid,0,1,p,good,colors)
            supported=[old.supported(p,good,k)and old.supported(p,alive,k)and old.supported(end,alive,k)for k in range(2)]
            y=[np.median(end[k,alive[k]]-p[k,alive[k]],axis=0)/320 if supported[k]else None for k in range(2)]
            u=None
            for obj in range(2):
                info=dict(object_index=obj,start_tick=previous['tick'],end_tick=tick,
                          segment=segment,dt_s=dt,initial_points=int(good[obj].sum()),
                          matched_points=int(alive[obj].sum()))
                if not supported[obj]:
                    missing.append(dict(info,reason='current_or_same_ID_correspondence_support_missing'));continue
                history=np.zeros(10)
                past=self.prior_labels.get(obj)
                if past is not None and past['end_tick']==previous['tick']:
                    pd=past['dt_s'];pc=past['command'];px=past['context']
                    grip0=px[[6,13]]*np.pi;grip1=pc[36:38]*.1
                    history=np.r_[past['target'].astype(float)/pd,pc[12:16]/pd,(grip1-grip0)/pd,pd,1.]
                context=old.context(obj,p,good,colors,start_command,self.robots,self.cfg,self.camera,np.zeros(10))
                context=np.r_[context,partner_features(p,good,colors,obj)]
                context[34:44]=history
                if u is None:u=old.action(start_command,window,self.robots,self.cfg,self.camera)
                partner=1-obj;relative=y[obj]-y[partner]if supported[partner]else np.full(2,np.nan)
                row=dict(info,context=context,command=u.copy(),target=y[obj],
                         start_points=p[obj].copy(),end_points=end[obj].copy(),point_alive=alive[obj].copy(),
                         relative_target=relative,relative_available=np.asarray(supported[partner],bool))
                rows.append(row)
        else:
            missing.append(dict(start_tick=previous['tick'],end_tick=tick,
                                reason='outside_fixed25_to125_actual_steps'))
        # Only completed arrived labels may be used at the NEXT window's start.
        for row in rows:self.prior_labels[row['object_index']]=row
        self.previous=now;self.command=next_command
        self.rows+=len(rows);self.missing+=len(missing)
        return rows,missing

    def finish(self):
        self.closed=True
        return dict(rows=self.rows,missing=self.missing,scope='No sampler/optimizer; caller proves actual tracker prefix.')
