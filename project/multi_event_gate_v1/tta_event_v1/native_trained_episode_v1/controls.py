"""Fixed wrong-feedback control: rotate only already-eligible own command summaries."""
import numpy as np

def wrong_commands(pool):
 rows=pool.rows();changed=dict(rows);records=[]
 for source,owner in pool.live.items():
  ids=pool.queue.eligible(owner.ticket)
  if not ids:continue
  mapping=dict(rows[source]);donors=ids[-1:]+ids[:-1];count=0
  for receiver,donor in zip(ids,donors,strict=True):
   value=rows[source][donor]['command']
   if not np.isfinite(value).all():raise ValueError('Actual finite past own commands required')
   mapping[receiver]=dict(rows[source][receiver],command=value.copy())
   count+=not np.array_equal(value,rows[source][receiver]['command'])
  changed[source]=mapping
  records.append(dict(source_query=source,receiver_rows=list(ids),donor_rows=list(donors),effective_changed_commands=count,control_noop=count==0,only_arrived_same_packet_own_commands=True))
 return changed,records
