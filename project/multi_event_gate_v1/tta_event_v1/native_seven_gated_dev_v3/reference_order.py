"""Only reorder required real episodes; never add compute or use outcome labels."""
def before_tail(branches):
 rows=list(branches)
 assert rows and rows[0]=='M_event_full'and len(set(rows))==len(rows)
 assert 'baseline'in rows and 'TTA'in rows
 rows.remove('TTA');rows.insert(rows.index('baseline'),'TTA')
 if 'handoff_reference'in rows:
  rows.remove('handoff_reference');rows.insert(1,'handoff_reference')
 return rows
