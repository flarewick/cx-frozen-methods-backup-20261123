"""Atomic whole-episode continuation with immutable evaluation identity."""
import hashlib
import json
from pathlib import Path
from .paths import output_path,replace_json
from .audit import digest


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_journal(run,manifest):
    path=run/'journal.json'
    state=json.loads(path.read_text()) if path.exists() else dict(identity=manifest['evaluation_identity'],attempts=[],episodes=[])
    if state['identity']!=manifest['evaluation_identity']:raise ValueError('Evaluation identity changed')
    a,e=state['attempts'],state['episodes'];start=manifest['seed_start']
    if len(a)>manifest['max_attempts'] or len(e)>manifest['episodes']:raise ValueError('Evaluation count overflow')
    accepted_index=0;by_seed={ep['seed']:ep for ep in e}
    if len(by_seed)!=len(e):raise ValueError('Duplicate completed episode seed')
    completed_seeds=[]
    for i,row in enumerate(a):
        if row['seed']!=start+i or row['episode_index']!=accepted_index or type(row['eligible'])is not bool:
            raise ValueError('Noncontiguous screening or accepted index')
        if row['eligible']:
            ep=by_seed.get(row['seed'])
            if ep is None:
                if i!=len(a)-1:raise ValueError('An interrupted policy episode was skipped')
                continue
            if ep['episode_index']!=accepted_index or ep.get('complete')is not True or type(ep.get('success'))is not bool:
                raise ValueError('Only complete Boolean policy outcomes can resume')
            if ep.get('condition')!=manifest['condition'] or ep.get('native_limit')!=manifest['official_task_action_budget']:
                raise ValueError('Condition/budget changed')
            if ep.get('initial_physical_sha256')!=row.get('initial_physical_sha256'):
                raise ValueError('Initial physical provenance changed')
            for name,sha in ep.get('evidence_sha256',{}).items():
                f=output_path(Path(name))
                if not f.is_relative_to(run) or digest(f)!=sha:raise ValueError('Completed episode evidence changed')
            if not ep.get('evidence_sha256') or not ep.get('process_session'):raise ValueError('Episode provenance missing')
            completed_seeds.append(ep['seed']);accepted_index+=1
    if completed_seeds!=[ep['seed'] for ep in e]:raise ValueError('Episodes do not follow screening order')
    return state


def stream(run,manifest,screen,rollout,update,announce):
    state=load_journal(run,manifest);a=state['attempts'];e=state['episodes']
    update(a,e)
    while len(e)<manifest['episodes']:
        if a and a[-1]['eligible'] and (not e or e[-1]['seed']!=a[-1]['seed']):
            row=a[-1]  # Keep eligibility; re-create the unfinished scene at same index.
        else:
            if len(a)>=manifest['max_attempts']:raise RuntimeError('Screening attempt limit exhausted')
            seed=manifest['seed_start']+len(a)
            row=screen(seed,len(e))
            if row['seed']!=seed or row['episode_index']!=len(e) or type(row['eligible'])is not bool:
                raise ValueError('Invalid eligibility record')
            a.append(row);replace_json(run/'journal.json',state);update(a,e)
        if not row['eligible']:continue
        ep=rollout(row['seed'],len(e),row)
        if (ep.get('complete')is not True or type(ep.get('success'))is not bool
            or ep.get('seed')!=row['seed'] or ep.get('episode_index')!=len(e)):
            raise ValueError('Truncated/invalid episode is not a completed trial')
        if (ep.get('condition')!=manifest['condition'] or ep.get('native_limit')!=manifest['official_task_action_budget']
            or ep.get('initial_physical_sha256')!=row.get('initial_physical_sha256')
            or not ep.get('process_session') or not ep.get('evidence_sha256')):
            raise ValueError('Invalid whole-episode provenance')
        for name,sha in ep['evidence_sha256'].items():
            f=output_path(Path(name))
            if not f.is_relative_to(run) or digest(f)!=sha:raise ValueError('Episode evidence missing or changed before commit')
        e.append(ep);replace_json(run/'journal.json',state);update(a,e)
        announce(ep,sum(v['success'] for v in e),len(e))
    return dict(status='complete',complete=True,attempted=len(a),completed=len(e),requested=manifest['episodes'])
