"""Full-policy query control only. Native terminal state never becomes feedback."""
import hashlib
import json
from numbers import Integral
import numpy as np
from native_phase_feedback_v1.feedback_queue import FeedbackQueue

def native_continue(task):
    """Same while guard as evaluate_stacking_adapter; no check_success invocation."""
    count, limit, done = task.take_action_cnt, task.step_lim, task.eval_success
    if (not isinstance(count, Integral) or isinstance(count, bool) or count < 0
            or not isinstance(limit, Integral) or isinstance(limit, bool) or limit <= 0
            or type(done) is not bool):
        raise ValueError('Native evaluator control state unavailable')
    return count < limit and not done

def image_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()

def boundary_proof(observed, *, query, tick):
    if type(query) is not int or query < 0 or type(tick) is not int or tick < 0:
        raise ValueError('Exact native query/tick required')
    if observed['observation_tick'] != tick:
        raise ValueError('Stale actor RGB at query boundary')
    rgb = observed['rgb']
    if len(rgb) != 3 or any(im.dtype != np.uint8 or im.shape != (240, 320, 3) for im in rgb):
        raise ValueError('Three exact current native RGB images required')
    state = observed['command_state']
    if state.dtype != np.float32 or state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Current own-command state required')
    value = dict(query=query, observation_tick=tick, RGB_sha256=[image_sha(im) for im in rgb],
                 own_command_state_sha256=image_sha(state),
                 scope='Arrived caller boundary before exactly one native draw; no outcome/model target.')
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return value, hashlib.sha256(encoded).hexdigest()

class ShadowBoundary:
    """No optimizer. Release once per source only when a legal real draw follows."""
    def __init__(self, fork, feedback):
        if not isinstance(feedback, FeedbackQueue):
            raise ValueError('Frozen phase-feedback queue required')
        self.fork, self.feedback = fork, feedback
        self.closed = False
        self.records = []

    def close(self):
        if not self.closed:
            self.final = self.feedback.finish()
            self.closed = True
        return dict(self.final)

    def draw(self, task, observed, operation, *, query, tick, delayed,
             device_uuid, model_sha256):
        if self.closed:
            raise ValueError('Closed episode boundary')
        # Driver control only: after terminal there is no feedback callback/draw.
        if not native_continue(task):
            self.close()
            return None
        proof, digest = boundary_proof(observed, query=query, tick=tick)
        if self.feedback.now != tick or self.feedback.frames[tick][1] != proof['RGB_sha256'][0]:
            raise ValueError('Feedback and actor boundary RGB differ')
        f = self.fork
        if delayed:
            f.permit_delayed(query=query, tick=tick, device_uuid=device_uuid, model_sha256=model_sha256)
            draw = f.draw_delayed
        else:
            f._identity(device_uuid, model_sha256)
            if not f.installed or f.following_draws:
                raise ValueError('Only first natural following query')
            f.ledger.require_following_query(tick)
            f.ledger.permit_query(query, tick)
            draw = f.draw_following
        packet = self.feedback.at_query(query=query, tick=tick, query_proof_sha256=digest)
        result, note = draw(operation, query=query, tick=tick,
                            device_uuid=device_uuid, model_sha256=model_sha256)
        # A failure in draw propagates and invalidates the attempt; never retry it.
        row = dict(boundary=proof, boundary_sha256=digest, actual_sampler=note,
                   released_rows=list(packet.rows) if packet else [],
                   source_query=packet.source_query if packet else None,
                   shadow_only=True, optimizer_updates=0)
        self.records.append(row)
        return result, note, packet
