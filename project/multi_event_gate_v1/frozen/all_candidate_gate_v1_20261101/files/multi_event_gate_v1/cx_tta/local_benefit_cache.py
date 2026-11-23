"""Compact all-query legal cache for future local-benefit collection.
Offline labels and episode outcomes never enter returned observable inputs.
"""
from pathlib import Path
import json
import numpy as np
from .paths import output_path,replace_json
from .structured_action_adapter import FORWARD_KEYS

def legal_packet(packet):
    if set(packet)!=FORWARD_KEYS|{'base'}:
        raise ValueError('Exactly nine legal fields plus actual current base required')
    shapes={'visual':(1,3,2,2,32),'text':(1,2,32),'state':(1,14),
        'past_visual':(1,3,2,2,32),'past_text':(1,2,32),'past_state':(1,14),
        'has_history':(1,1),'ee_pose':(1,2,9),'past_ee_pose':(1,2,9),'base':(1,32,14)}
    for k,v in packet.items():
        if not isinstance(v,np.ndarray) or v.dtype!=np.float32 or v.shape!=shapes[k] or not np.isfinite(v).all():
            raise ValueError('Finite nativefloat32 observable shape: '+k)
    if packet['has_history'][0,0] not in (0,1):raise ValueError('Binary realhistory flag')
    return {k:v.copy() for k,v in packet.items()}

class AllQueryCache:
    def __init__(self,folder):
        self.folder=output_path(Path(folder));self.folder.mkdir(exist_ok=False)
        self.packets=[];self.raw=[];self.applied=[];self.actions=[];self.valid=[];self.labels=[];self.rows=[];self.closed=False
    def append(self,query,packet,raw_delta,applied_delta,actions,executed,offline_labels):
        if self.closed or type(query)is not int or query!=len(self.packets):
            raise ValueError('Open contiguous query stream required')
        p=legal_packet(packet);previous=self.packets[-1] if query else None
        if bool(p['has_history'][0,0])!=(previous is not None):raise ValueError('Actual history flag')
        for k in ('visual','text','state','ee_pose'):
            want=previous[k] if previous is not None else np.zeros_like(p['past_'+k])
            if p['past_'+k].tobytes()!=want.tobytes():raise ValueError('History differs from actual preceding query: '+k)
        values=[np.asarray(v) for v in (raw_delta,applied_delta,actions)]
        if any(v.shape!=(32,14) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in values):
            raise ValueError('Actual nativefloat32 command/proposal shapes')
        raw,delta,act=values
        if type(executed)is not int or not 0<executed<=32:raise ValueError('True executed prefix1..32,not observation callbacks')
        base=p['base'][0];off=delta==0
        if act[off].tobytes()!=base[off].tobytes():raise ValueError('Disabled coordinates must preserve original base bytes')
        if not np.array_equal(act[~off],(base+delta)[~off]):raise ValueError('Applied command formula')
        label_snapshot=json.loads(json.dumps(offline_labels,allow_nan=False))
        self.packets.append(p);self.raw.append(raw.copy());self.applied.append(delta.copy());self.actions.append(act.copy())
        self.valid.append(np.arange(32)<executed);self.labels.append(label_snapshot)
        self.rows.append(dict(query=query,executed=executed,previous_query=query-1 if query else None))
        self._save()
    def _save(self):
        values={k:np.concatenate([p[k] for p in self.packets],axis=0) for k in FORWARD_KEYS|{'base'}}
        values.update(raw_delta=np.stack(self.raw),applied_delta=np.stack(self.applied),actions=np.stack(self.actions),valid=np.stack(self.valid))
        file=self.folder/'queries.npz';tmp=file.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,**values)
        tmp.replace(file)
        replace_json(self.folder/'offline_labels.json',dict(schema=1,scope='Offline only; no query benefit label inferred from episode success',queries=self.labels))
        replace_json(self.folder/'manifest.json',dict(schema=1,status='collecting',queries=len(self.rows),rows=self.rows,
            forward_keys=sorted(FORWARD_KEYS),base_separate=True,no_video_or_RGB=True,all_actual_queries=True,training_approved=False))
    def finish(self,*,complete,success,role):
        if self.closed or not self.packets:raise ValueError('Nonempty open stream required')
        if type(complete)is not bool or (complete and type(success)is not bool) or (not complete and success is not None):
            raise ValueError('Censored outcomes must be None')
        if role not in ('diagnostic_confirmation','independent_train_candidate'):raise ValueError('Explicit provenance role required')
        m=json.loads((self.folder/'manifest.json').read_text());m.update(status='complete' if complete else 'censored',
            role=role,episode_success=success,training_approved=False,episode_outcome_not_perquery_label=True)
        replace_json(self.folder/'manifest.json',m);self.closed=True

def load_observables(folder):
    folder=output_path(Path(folder));m=json.loads((folder/'manifest.json').read_text())
    with np.load(folder/'queries.npz',allow_pickle=False) as z:
        n=len(z['base']);x={k:z[k].copy() for k in FORWARD_KEYS};base=z['base'].copy()
        required=FORWARD_KEYS|{'base','raw_delta','applied_delta','actions','valid'}
        if set(z.files)!=required or n!=m['queries'] or set(m['forward_keys'])!=FORWARD_KEYS:
            raise ValueError('Stored schema/count mismatch')
    # Deliberately do not read offline_labels.json or copy manifest metadata to x.
    for i in range(n):legal_packet({**{k:v[i:i+1] for k,v in x.items()},'base':base[i:i+1]})
    return x,base
