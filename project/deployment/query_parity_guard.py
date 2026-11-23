"""Compare completed whole episodes with a retained independent reference."""
import gzip
import hashlib
import json
import zlib
from pathlib import Path


def rows(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream]


def values(value):
    if isinstance(value,dict):return {k:values(v) for k,v in value.items() if k!='feature_seconds'}
    if isinstance(value,list):return [values(v) for v in value]
    return value


def query_file(run,episode):
    session=episode['process_session']
    if not session.startswith('process_') or '/' in session:raise ValueError('Unsafe parity session')
    paths=list((run/'sessions'/session/'queries').glob('*_seed'+str(episode['seed'])+'.jsonl.gz'))
    if len(paths)!=1:raise ValueError('Expected exactly one episode query file')
    return paths[0]


class QueryParityGuard:
    def __init__(self,reference):
        self.reference=Path(reference).resolve()
        state=json.loads((self.reference/'journal.json').read_text())
        self.episodes={ep['seed']:ep for ep in state['episodes']}
        if not self.episodes:raise ValueError('No complete reference episodes')
        self.expected={seed:rows(query_file(self.reference,ep)) for seed,ep in self.episodes.items()}
        self.checked={};self.partial_checked={}

    def check_partial(self,run):
        run=Path(run)
        for path in sorted((run/'sessions').glob('process_*/queries/*_seed*.jsonl.gz')):
            seed=int(path.name.split('_seed')[-1].split('.')[0])
            if seed not in self.expected:continue
            try:data=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(path.read_bytes())
            except zlib.error:return 'Invalid streaming query evidence at '+str(path)
            lines=data.split(b'\n')[:-1]
            actual=[json.loads(line)for line in lines if line.strip()]
            if not actual:continue
            if values(actual)!=values(self.expected[seed][:len(actual)]):return 'Concurrent streaming query parity failed at seed '+str(seed)+' after '+str(len(actual))+' queries'
            self.partial_checked[str(path)]=dict(seed=seed,matching_queries=len(actual),exact_except_feature_seconds=True)
        if self.partial_checked:
            target=run/'deployment_query_parity_partial.json'
            temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(reference=str(self.reference),matching_prefixes=self.partial_checked,partial_queries_are_not_complete_episodes=True,results_merged=False),indent=2)+'\n');temporary.replace(target)
        return None

    def check(self,run):
        run=Path(run)
        partial_error=self.check_partial(run)
        if partial_error:return partial_error
        journal=run/'journal.json'
        if not journal.exists():return None
        episodes={ep['seed']:ep for ep in json.loads(journal.read_text())['episodes']}
        queue=run/'concurrency_queue.json'
        if queue.exists():
            for ep in json.loads(queue.read_text())['completed'].values():
                if ep['seed']in episodes and episodes[ep['seed']]!=ep:return 'Queue and committed parity episodes diverged'
                episodes[ep['seed']]=ep
        for episode in sorted(episodes.values(),key=lambda ep:ep['episode_index']):
            seed=episode['seed']
            if seed not in self.expected or seed in self.checked:continue
            path=query_file(run,episode)
            actual=rows(path);expected=self.expected[seed]
            if values(actual)!=values(expected):
                return 'Observation optimization query parity failed at seed '+str(seed)
            reference=self.episodes[seed]
            if any(episode.get(k)!=reference.get(k) for k in ('success','executed_actions','queries')):
                return 'Observation optimization episode outcome parity failed at seed '+str(seed)
            self.checked[seed]=dict(queries=len(actual),exact_except_feature_seconds=True,
                source_sha256=hashlib.sha256(query_file(self.reference,reference).read_bytes()).hexdigest(),
                optimized_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            result=dict(reference=str(self.reference),optimized_run=str(run),episodes=self.checked,
                        all_available_reference_episodes_checked=len(self.checked)==len(self.expected),
                        observations_rng_actions_exact=True,results_merged=False)
            (run/'deployment_query_parity.json').write_text(json.dumps(result,indent=2)+'\n')
            print('\nEVAL_QUERY_PARITY_PASSED',seed,len(actual),flush=True)
        return None
