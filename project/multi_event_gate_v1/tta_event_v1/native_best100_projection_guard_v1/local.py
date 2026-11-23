"""Missing-feedback handling for the calibrated commanded-TCP domain boundary."""
import numpy as np
from native_delayed_query_v1.local_feedback import LocalFeedbackStream as Original
from native_delayed_query_v1.primitives import old

PROJECTION_ERROR = 'Commanded point outside calibrated forward frustum'

class LocalFeedbackStream(Original):
 def frame(self, *args, **kwargs):
  value = super().frame(*args, **kwargs)
  if getattr(self, '_capture_arrival', False): self._arrived_frame = value
  return value

 def arrive(self, rgb, masks, presence, trace_window, command_state, *, tick, elapsed_s, segment, segment_start_tick):
  previous = self.previous
  command = self.command.copy()
  counters = (self.rows, self.missing)
  labels = dict(self.prior_labels)
  self._capture_arrival = True
  self._arrived_frame = None
  try:
   return super().arrive(rgb, masks, presence, trace_window, command_state,
       tick=tick, elapsed_s=elapsed_s, segment=segment, segment_start_tick=segment_start_tick)
  except ValueError as error:
   if str(error) != PROJECTION_ERROR: raise
   now = self._arrived_frame
   # Original validation of the complete arrived trace/state has already passed.
   # The original implementation commits no state until all geometry succeeds.
   assert now is not None and now['tick'] == tick and 25 <= tick-previous['tick'] <= 125
   assert self.previous is previous and np.array_equal(self.command, command)
   assert (self.rows, self.missing) == counters
   assert self.prior_labels.keys() == labels.keys()
   assert all(self.prior_labels[k] is labels[k] for k in labels)
   next_command = command.copy()
   next_command[old.ARM] = np.asarray(trace_window['arm_position_target'])[-1]
   next_command[old.GRIP] = np.asarray(trace_window['gripper_logical_command'])[-1]
   assert np.array_equal(next_command.astype(np.float32), np.asarray(command_state, np.float32))
   missing = dict(start_tick=previous['tick'], end_tick=tick, segment=segment,
       reason='commanded_TCP_projection_unsupported', no_target_generated=True,
       calibrated_projection_unchanged=True)
   self.previous = now
   self.command = next_command
   self.missing += 1
   # Keep old labels. Their end_tick will fail the existing contiguous-history
   # check after this gap; no invalid label or replacement motion is invented.
   return [], [missing]
  finally:
   self._capture_arrival = False
   self._arrived_frame = None
