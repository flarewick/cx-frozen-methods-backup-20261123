"""Durable out-of-order whole episodes with an original ordered journal prefix.

The queue is a write-ahead record. Complete evidence is validated before it is
saved; original journals remain readable by the unchanged original validator.
Unfinished scenes retain their exact eligible seed/index after interruption.
"""
import copy
import json
import threading
from pathlib import Path


class EpisodeQueue:
    def __init__(self,run,manifest,update=lambda *a:None,announce=lambda *a:None):
        from cx_tta.evaluation_resume import load_journal
        self.run=Path(run);self.manifest=manifest;self.path=self.run/'concurrency_queue.json'
        self.lock=threading.Condition(threading.RLock());self.claimed=set();self.screening=False
        self.update=update;self.announce=announce
        journal=load_journal(self.run,manifest)
        if self.path.exists():self.state=json.loads(self.path.read_text())
        else:self.state=dict(identity=manifest['evaluation_identity'],attempts=copy.deepcopy(journal['attempts']),completed={str(e['seed']):copy.deepcopy(e) for e in journal['episodes']})
        self.validate()
        if journal['attempts']!=self.state['attempts'][:len(journal['attempts'])]:raise ValueError('Queue and original screening prefix diverged')
        for ep in journal['episodes']:
            if self.state['completed'].get(str(ep['seed']))!=ep:raise ValueError('Queue and original completed prefix diverged')
        self.journal=journal
        # Reconcile a crash between durable queue save and original journal save.
        self._persist();self._commit()

    def _episode(self,ep,row):
        from cx_tta.paths import output_path
        from cx_tta.audit import digest
        m=self.manifest
        if (ep.get('complete')is not True or type(ep.get('success'))is not bool or ep.get('seed')!=row['seed'] or ep.get('episode_index')!=row['episode_index'] or ep.get('condition')!=m['condition'] or ep.get('native_limit')!=m['official_task_action_budget'] or ep.get('initial_physical_sha256')!=row.get('initial_physical_sha256') or not ep.get('process_session') or not ep.get('evidence_sha256')):
            raise ValueError('Invalid completed queue episode/provenance')
        for name,sha in ep['evidence_sha256'].items():
            f=output_path(Path(name))
            if not f.is_relative_to(self.run) or digest(f)!=sha:raise ValueError('Queue evidence missing, changed or outside dataset')

    def validate(self):
        s=self.state;m=self.manifest
        if s['identity']!=m['evaluation_identity']:raise ValueError('Queue identity changed')
        if len(s['attempts'])>m['max_attempts']:raise ValueError('Queue attempt overflow')
        index=0;eligible={}
        for i,row in enumerate(s['attempts']):
            if row['seed']!=m['seed_start']+i or row['episode_index']!=index or type(row['eligible'])is not bool:raise ValueError('Queue seeds/indices are not contiguous')
            if row['eligible']:eligible[str(row['seed'])]=row;index+=1
        if index>m['episodes']:raise ValueError('Queue eligible episode overflow')
        if not set(s['completed'])<=set(eligible):raise ValueError('Queue outcome has no screened eligible scene')
        for key,ep in s['completed'].items():self._episode(ep,eligible[key])

    def _persist(self):
        from cx_tta.paths import replace_json
        replace_json(self.path,self.state)

    def _commit(self):
        from cx_tta.paths import replace_json
        from cx_tta.evaluation_resume import load_journal
        attempts=[];episodes=[]
        for row in self.state['attempts']:
            attempts.append(copy.deepcopy(row))
            if row['eligible']:
                ep=self.state['completed'].get(str(row['seed']))
                if ep is None:break
                episodes.append(copy.deepcopy(ep))
        old=len(self.journal['episodes'])
        journal=dict(identity=self.manifest['evaluation_identity'],attempts=attempts,episodes=episodes)
        replace_json(self.run/'journal.json',journal)
        # Fail closed using the original evidence/order validator, including
        # after recovery of a fully saved out-of-order episode.
        self.journal=load_journal(self.run,self.manifest)
        self.update(copy.deepcopy(attempts),copy.deepcopy(episodes))
        for i,ep in enumerate(episodes[old:],old):self.announce(ep,sum(x['success'] for x in episodes[:i+1]),i+1)

    def acquire(self,screen,cancel):
        with self.lock:
            while True:
                if cancel.is_set():raise RuntimeError('Queue cancelled by peer/supervisor')
                for row in self.state['attempts']:
                    if row['eligible'] and str(row['seed'])not in self.state['completed'] and row['seed']not in self.claimed:
                        self.claimed.add(row['seed']);return copy.deepcopy(row)
                accepted=sum(row['eligible'] for row in self.state['attempts'])
                if accepted==self.manifest['episodes']:return None
                if self.screening:self.lock.wait(.1);continue
                if len(self.state['attempts'])>=self.manifest['max_attempts']:raise RuntimeError('Screening attempt limit exhausted')
                seed=self.manifest['seed_start']+len(self.state['attempts']);index=accepted;self.screening=True
                # Never hold the condition lock while waiting for the model
                # owner service; peers must still be able to commit results.
                self.lock.release()
                try:row=screen(seed,index)
                finally:self.lock.acquire();self.screening=False;self.lock.notify_all()
                if row['seed']!=seed or row['episode_index']!=index or type(row['eligible'])is not bool:raise ValueError('Invalid original screening row')
                self.state['attempts'].append(copy.deepcopy(row));self._persist();self._commit()

    def complete(self,row,episode):
        with self.lock:
            if row['seed']not in self.claimed:raise ValueError('Scene was not assigned to this queue')
            self._episode(episode,row)
            key=str(row['seed'])
            if key in self.state['completed']:raise ValueError('Cannot replace a completed episode')
            self.state['completed'][key]=copy.deepcopy(episode)
            self._persist();self._commit();self.claimed.remove(row['seed']);self.lock.notify_all()

    def work(self,callbacks,cancel):
        while True:
            row=self.acquire(callbacks['screen'],cancel)
            if row is None:return
            episode=callbacks['rollout'](row['seed'],row['episode_index'],row)
            self.complete(row,episode)
