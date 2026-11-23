"""Reject alternate/inactive/later target choices before loading any training tensors."""
from .data import SelectedReader,T,read
class QualifiedSelectedReader(SelectedReader):
 def __init__(self):
  super().__init__();report=read(T/'outputs/native_outer_alignment_audit_v1/report.json')
  assert report['passed']and report['packets_retained']==261
  self.targets={(r['case_id'],r['source_query'],r['feedback_query']):r['next_selected_at_or_after_feedback']for r in report['packets']}
 def check_selection(self,p,selected_query):
  key=(p['case_id'],p['source_query'],p['query'])
  if key not in self.targets:raise ValueError('Packet outside the preserved original261 audit')
  expected=self.targets[key]
  if expected is None or not expected['valid_command_count']:raise ValueError('No executable outer target; preserve packet without training loss')
  if type(selected_query)is not int or selected_query!=expected['query']:raise ValueError('Only the first actual selected query at/after feedback is an eligible outer target')
  return expected
 def packet_for_action(self,p,queries,selected_query):
  self.check_selection(p,selected_query)
  return super().packet_for_action(p,queries,selected_query)
