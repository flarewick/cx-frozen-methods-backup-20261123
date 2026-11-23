"""Real eligible command rows: deterministic wrong control, no label/future change."""
from pathlib import Path
from types import SimpleNamespace
import json,numpy as np
from .controls import wrong_commands
from native_policy_execution_v1.common import T,read,sha,write

def main():
 packets=read(T/'outputs/native_expert_packets_v1/manifest.json')['packets'];records=[];total=0
 for p in packets:
  with np.load(p['local_file'],allow_pickle=False)as z:local={k:z[k]for k in z.files}
  ids=tuple(p['row_ids']);source=p['source_query'];rows={i:{k:local[k][i].copy()for k in ('context','command','target','start_points')}for i in ids}
  original={i:{k:v.tobytes()for k,v in row.items()}for i,row in rows.items()}
  pool=SimpleNamespace(rows=lambda:{source:rows},live={source:SimpleNamespace(ticket='actual_eligible_fixture')},queue=SimpleNamespace(eligible=lambda ticket:ids))
  transformed,metadata=wrong_commands(pool);assert len(metadata)==1;record=metadata[0]
  expected_donors=ids[-1:]+ids[:-1];assert record['donor_rows']==list(expected_donors)
  for recipient,donor in zip(ids,expected_donors,strict=True):
   assert transformed[source][recipient]['command'].tobytes()==rows[donor]['command'].tobytes()
   for field in ('context','target','start_points'):assert transformed[source][recipient][field].tobytes()==original[recipient][field]
  assert all(row[k].tobytes()==original[i][k]for i,row in rows.items()for k in row)
  total+=record['effective_changed_commands'];records.append(dict(case_id=p['case_id'],query=p['query'],**record))
  pool.queue.eligible=lambda ticket:(ids[0],)
  single,meta=wrong_commands(pool);assert meta[0]['control_noop']and meta[0]['effective_changed_commands']==0 and single[source][ids[0]]['command'].tobytes()==original[ids[0]]['command']
  pool.queue.eligible=lambda ticket:()
  empty,meta=wrong_commands(pool);assert not meta and empty[source]is rows
 assert len(records)==12 and total>0
 write(T/'outputs/native_trained_episode_v1/CPU_wrong_feedback_control_checks.json',dict(passed=True,real_old_packets=12,changed_command_summaries=total,records=records,only_eligible_own_history_permuted=True,context_target_startpoints_and_original_rows_unchanged=True,single_row_noop_retained=True,empty_packet_not_replaced=True,new_native_rollouts=0,actual_actor_updates=0,source_sha256={str(p):sha(p)for p in [Path(__file__),Path(__file__).with_name('controls.py')]}))
 print('FIXED_WRONG_FEEDBACK_ELIGIBLE_COMMAND_CONTROL_PASS',12,total,flush=True)
if __name__=='__main__':main()
