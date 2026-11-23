"""Validate every original identity and its raw missing/available provenance."""
POLICY='all_original100_attempted_and_audited_unknowns_zero_loss_v1'

def validate(original,index,coverage,packets,read,sha):
 expected={r['case_id']:r for r in original['records']}
 cases={r['case_id']:r for r in index['cases']};rows={r['case_id']:r for r in coverage}
 assert len(expected)==len(cases)==len(index['cases'])==len(rows)==len(coverage)==100
 assert set(expected)==set(cases)==set(rows)
 used=set();counts={}
 for key,c in cases.items():
  r=expected[key];row=rows[key]
  for field,expected_value in [('original_expert_index',r['index']),('task',r['task']),('split',r['split'])]:assert c[field]==row[field]==expected_value
  assert row['feedback_processing_complete']and row['native_available']==c['native_available']and row['status']==c['feedback_processing_status']
  group=c['task']+'/'+c['split'];counts[group]=counts.get(group,0)+1
  evidence=c['native_evidence'];assert sha(evidence['file'])==evidence['sha256']
  native=read(evidence['file'])['rows'][evidence['row_index']]
  assert native['case_id']==key and native['available']==c['native_available']
  own=[p for p in packets if p['case_id']==key]
  assert len(own)==row['packets']<=3
  if not c['native_available']:
   assert not own and row['processed_streams']==0
   assert c['native_missing_reason']==row['native_missing_reason']==native['reason']
  elif not c['seed']['available']:
   assert not own and row['processed_streams']==0 and row['semantic_reason']==c['seed']['reason']
  elif not c['old_pilot_reused']:
   assert row['processed_streams']==sum(j['case_id']==key for j in index['jobs'])
  else:assert row['processed_streams']==3
 assert counts==original['counts']and sorted(counts.values())==[10,10,40,40]
 for p in packets:
  assert p['case_id']in cases
  c=cases[p['case_id']];assert c['native_available']and c['seed']['available']
  assert p['original_expert_index']==c['original_expert_index']and p['split']==c['split']
  pair=(p['case_id'],p['source_query']);assert pair not in used;used.add(pair)
 return dict(passed=True,counts=counts,requested=100,native_available=sum(c['native_available']for c in cases.values()),semantic_available=sum(c['native_available']and c['seed']['available']for c in cases.values()),policy=POLICY)
