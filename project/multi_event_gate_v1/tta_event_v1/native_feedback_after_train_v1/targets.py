"""First structurally legal application of ONE arrived feedback revision."""
def first_action(queries,packet,config,*,other_packets=()):
 source=packet['source_query'];arrival=packet['query']
 if arrival not in(source+1,source+2):raise ValueError('Outside original causal feedback horizon')
 if packet.get('single_use')is not True:raise ValueError('Actual single-use packet required')
 superseding=min((p['query']for p in other_packets if p['query']>arrival),default=arrival+3)
 last=None
 for row in queries:
  q=row['query'];decision=row['event']['decision'];selected=decision['active']and decision['selected_event']is not None
  if selected:last=q
  if not arrival<=q<=arrival+2 or q>=superseding:continue
  if not row['target']['valid_command_count']:continue
  if selected:return dict(query=q,mode='original_selected',zero_residual='reference',hypothetical_eligibility_only=True)
  if decision['interventions']>=config['max_interventions']:continue
  if last is not None and q-last<=config['cooldown_queries']:continue
  return dict(query=q,mode='feedback_fast_difference',zero_residual='exact_zero',hypothetical_eligibility_only=True)
 return None
