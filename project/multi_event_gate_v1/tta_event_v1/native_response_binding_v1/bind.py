"""Bind arrived local response to its real issuing hidden and current query."""
import numpy as np
import torch
ARM=np.array([0,1,2,3,4,5,7,8,9,10,11,12],np.int64)
def tensor(value):return torch.as_tensor(np.array(value,copy=True),dtype=torch.float32)

def inputs(packet,*,rows,issuing_queries,current,predictor):
 ids=tuple(packet.rows)
 if not ids or len(ids)!=len(set(ids)):raise ValueError('Exact nonempty single-use packet rows required')
 if current['query']!=packet.query or current['tick']!=packet.observation_tick or current['query_proof_sha256']!=packet.query_proof_sha256:raise ValueError('Feedback/current-query proof mismatch')
 if packet.query not in(packet.source_query+1,packet.source_query+2):raise ValueError('Frozen source origin+following horizon required')
 selected=[];hidden=[]
 for i in ids:
  row=rows[i];q=row['command_query']
  if q not in(packet.source_query,packet.source_query+1)or q>=packet.query:raise ValueError('Future or wrong issuing command query')
  if not 0<=row['start_tick']<row['end_tick']<=packet.observation_tick:raise ValueError('Local response has not actually arrived')
  owner=issuing_queries[q]
  if owner['query']!=q or owner['tick']>row['start_tick']:raise ValueError('Hidden state does not belong to actual issuing query')
  value=np.asarray(owner['hidden'])
  if value.shape!=(32,64)or not np.isfinite(value).all():raise ValueError('Actual issuing hidden64 required')
  selected.append({k:row[k]for k in('context','command','start_points','target')});hidden.append(value)
 x=np.stack([r['context']for r in selected]);u=np.stack([r['command']for r in selected]);points=np.stack([r['start_points']for r in selected]);target=np.stack([r['target']for r in selected])
 if x.shape!=(len(ids),53)or u.shape!=(len(ids),197)or points.shape!=(len(ids),24,2)or target.shape!=(len(ids),2):raise ValueError('Original local feedback feature shapes required')
 if any(not np.isfinite(v).all()for v in (x,u,target)):raise ValueError('Finite arrived local feedback required')
 xx,command,kind,_=predictor.model.coordinates(x,u,points,'correct');prediction=predictor.model.predict(predictor.state,xx,command,kind)
 h=np.asarray(current['hidden']);logits=np.asarray(current['logits']);reference=np.asarray(current['raw_delta'])
 if h.shape!=(32,64)or logits.shape!=(32,14)or reference.shape!=(32,14)or any(not np.isfinite(v).all()for v in(h,logits,reference)):raise ValueError('Actual current hidden/logits/reference required')
 return dict(row_ids=ids,hidden=tensor(np.stack(hidden)),context=tensor(xx),command=tensor(command),kind=tensor(kind),kernel_prediction=tensor(prediction),target=tensor(target),next_hidden=tensor(h[None]),next_logits=tensor(logits[:,ARM][None]),next_reference=tensor(reference[:,ARM][None]))
