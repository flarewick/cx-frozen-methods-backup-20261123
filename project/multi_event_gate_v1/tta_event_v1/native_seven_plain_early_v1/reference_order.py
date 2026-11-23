"""Run required whole full reference early; no episode is added or omitted."""
def before_tail(branches):
 rows=list(branches);assert rows and rows[0]=='M_event_full'and len(set(rows))==len(rows)
 assert 'TTA'in rows and 'baseline'not in rows
 if 'handoff_reference'in rows:rows.remove('handoff_reference');rows.insert(1,'handoff_reference')
 return rows

def include_reference(branches):
 return before_tail(list(branches)+([]if'handoff_reference'in branches else['handoff_reference']))
