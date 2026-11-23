"""Real selected rows serialize losslessly despite whole-stream NaN promotion."""
from pathlib import Path
import json,numpy as np
from .export import arrays,FIELDS
from native_policy_execution_v1.common import T,read,sha,write

def main():
 old=read(T/'outputs/native_expert_packets_v1/manifest.json');p=next(p for p in old['packets']if p['case_id']=='stack_bowls_two:7100000'and p['query']==p['source_query']+1)
 with np.load(p['local_file'],allow_pickle=False)as z:source={k:z[k]for k in z.files}
 rows=[]
 for i in p['row_ids']:
  row={k:source[k][i].copy()for k in FIELDS};row['relative_target']=row['relative_target'].astype(np.float32)
  for k in ('command_query','start_tick','end_tick'):row[k]=int(source[k][i])
  rows.append(row)
 actual=arrays(rows)
 for key in FIELDS+('command_query','start_tick','end_tick'):
  expected=source[key][p['row_ids']]
  if key=='relative_target':expected=expected.astype(np.float64)
  assert actual[key].dtype==expected.dtype and actual[key].shape==expected.shape and actual[key].tobytes()==expected.tobytes(),key
 assert all(r['relative_target'].dtype==np.float32 for r in rows)
 write(T/'outputs/native_live_response_source_v1/CPU_causal_source_serialization_checks.json',dict(passed=True,actual_saved_packet=[p['case_id'],p['source_query'],p['query']],primary_feature_bytes_exact=True,relative_diagnostic_float64_lossless=True,input_rows_not_mutated=True,new_native_rollouts=0,GPU_forwards=0,source_sha256={str(f):sha(f)for f in [Path(__file__),Path(__file__).with_name('export.py')]}))
 print('CAUSAL_SOURCE_SERIALIZATION_LOSSLESS_PASS',len(rows),flush=True)
if __name__=='__main__':main()
