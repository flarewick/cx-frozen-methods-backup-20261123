"""All-phase frozen diagnostic, never a post-hoc candidate promotion or refit."""
from pathlib import Path
import importlib.util,numpy as np
C=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('actual_structured_phase_diagnostic_study',C/'study.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
def run():
 p,x,a,meta=s.dataset();T=s.T;D=s.D;pred=s.load(D/'predictions.npz');report=s.read(D/'report.json')
 spec=s.read(T/'outputs/native_arrived_mask_v3/protocol.json');jobs={j['index']:j for j in spec['jobs']}
 groups={};following_receipts=[];checks=0
 for i,r in enumerate(meta):
  job=jobs[r['stream_index']];origin=job['stream']['segments'][0];tick=origin['previous_origin_requested95_absolute_tick']
  assert tick is not None and origin['annotation_available_tick']==r['origin_receipt_tick'];checks+=1
  if r['segment']=='origin':
   phase='origin_pre95'if r['end_tick']<tick else'origin_cross95'if r['start_tick']<tick else'origin_after95'
  else:phase='following'
  groups.setdefault(phase,[]).append(i)
 result={}
 for phase in('origin_pre95','origin_cross95','origin_after95','following'):
  ids=np.array(groups.get(phase,[]),int)
  result[phase]=dict(rows=len(ids),cases=len({meta[i]['case_id']for i in ids}))
  if len(ids):
   result[phase]['frozen_correct']=s.metrics.summary(pred['correct'][ids],a['target'][ids],[meta[i]for i in ids])
 for job in spec['jobs']:
  origin=job['stream']['segments'][0];following=[v for v in job['stream']['segments']if v['kind']=='following']
  eligible=bool(job['origin_feedback_ready']and len(following)==1 and following[0]['complete_native32'])
  following_receipts.append(dict(case_id=job['case_id'],branch=job['branch'],full_origin_ready=job['origin_feedback_ready'],
    following_exists=bool(following),following_complete_native32=following[0]['complete_native32']if following else False,
    hypothetical_delayed_update_tick=following[0]['annotation_available_tick']if eligible else None,
    next_query_after_following_NOT_collected=True))
 out=dict(complete=True,checks=checks,all_rows=len(meta),strata=result,following_receipts=following_receipts,
  original_requested_cases=32,missing_cases=p['missing_cases'],
  full_following_receipt_streams=sum(r['hypothetical_delayed_update_tick']is not None for r in following_receipts),
  original_support_gate_failed_preserved=not report['diagnostic_support_gate_passed'],
  scope='All pre/cross/post95/following reported. Descriptive phase diagnosis only, no favorable stratum promoted, no refit, no online delay candidate implemented, no third query/task benefit.',
  source_sha256=s.sha(Path(__file__)),input_sha256={str(D/'report.json'):s.sha(D/'report.json'),str(D/'predictions.npz'):s.sha(D/'predictions.npz')},
  online_updates=0,TTA_utility_proven=False)
 s.write(s.O/'phase_diagnosis.json',out);print('FROZEN_ALL_PHASE_DIAGNOSIS', {k:v['rows']for k,v in result.items()},out['full_following_receipt_streams'],flush=True)
if __name__=='__main__':run()
