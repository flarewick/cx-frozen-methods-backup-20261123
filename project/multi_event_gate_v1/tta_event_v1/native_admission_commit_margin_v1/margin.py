"""Additional CPU-only wait margin; original final strict admission stays unchanged."""
def require(proof, selected_uuid):
 assert proof['passed']
 fleet=proof['latest_observation']['fleet'];estimates=proof['estimates']
 for card in fleet['cards']:
  if card['uuid']==selected_uuid or not card['own_pids']:continue
  v=estimates[card['uuid']]
  assert v['eligible']and v['unobserved_prefix_credit']==0
  # No future use is credited: require the original55% even with eight more
  # unobserved seconds, before starting the short CPU guard-setup interval.
  if v['observed_memory_time_integral']/(v['process_age_seconds']+8)<.55:
   raise RuntimeError('CPU wait: original55% needs eight-second unknown-zero commit margin')
 return proof
