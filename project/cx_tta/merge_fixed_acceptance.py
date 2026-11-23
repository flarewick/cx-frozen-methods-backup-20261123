"""Consolidate complete switch and corrected shoe acceptance without new sampling."""
import argparse,json
from pathlib import Path
from .audit import digest
from .cache_recovery import read,verify_hashes
from .paths import ROOT,new_run,write_json
from .audit_fixed_support import audit as support_audit

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--switch-stage',type=Path,required=True);ap.add_argument('--shoe-stage',type=Path,required=True);ap.add_argument('--held-stage',type=Path);a=ap.parse_args()
    stages=[a.switch_stage.resolve(),a.shoe_stage.resolve()];protocols=[read(x/'protocol.json') for x in stages];batches=[read(x/'batch_result.json') for x in stages]
    for p,b in zip(protocols,batches):
        verify_hashes({**p['source_sha256'],**p['code_sha256']})
        if b['status']!='complete':raise ValueError('Incomplete batch')
    p=protocols[0];attempts=[r for r in batches[0]['attempts'] if r['kind']=='switch']+[r for r in batches[1]['attempts'] if r['kind']=='shoe_support']
    if sorted(r['point'] for r in attempts)!=[0,1,2,3,*range(5,16)]:raise ValueError('Missing/duplicate fixed attempts')
    for record in attempts:
        path=Path(record['run'])
        if record['exit_code']:raise ValueError('Unresolved failed interface')
        if record['kind']=='shoe_support':
            if not (path/'audit.json').exists():support_audit(path)
            if not read(path/'audit.json')['passed']:raise ValueError('Support record audit failed')
        for f in path.glob('*.json'):
            if f.name not in ('progress.json',):p['source_sha256'][str(f)]=digest(f)
    for stage,other in zip(stages,protocols):
        p['source_sha256'].update(other['source_sha256']);p['code_sha256'].update(other['code_sha256'])
        for name in ('protocol.json','batch_result.json'):p['source_sha256'][str(stage/name)]=digest(stage/name)
    held_attempts=[]
    qualified_support={r['point'] for r in attempts if r['kind']=='shoe_support' and read(Path(r['run'])/'result.json')['held_qualified']}
    if qualified_support and not a.held_stage:raise ValueError('Complete qualified held forks before cache')
    if a.held_stage:
        stage=a.held_stage.resolve();hp=read(stage/'protocol.json');hr=read(stage/'result.json')
        if hr['status']!='complete':raise ValueError('Held stage incomplete')
        held_attempts=hr['attempts']
        if {r['point'] for r in held_attempts}!=qualified_support or any(r['exit_code'] for r in held_attempts):raise ValueError('Held coverage mismatch')
        verify_hashes({**hp['source_sha256'],**hp['code_sha256']})
        p['source_sha256'].update(hp['source_sha256']);p['code_sha256'].update(hp['code_sha256'])
        for f in [stage/'protocol.json',stage/'result.json']:
            p['source_sha256'][str(f)]=digest(f)
        for record in held_attempts:
            for f in Path(record['run']).glob('*.json'):
                if f.name!='progress.json':p['source_sha256'][str(f)]=digest(f)
    p.update(version='S2d-fixed-acceptance-consolidated',status='all_fixed_points_attempted_before_cache',
        acceptance_stages=[str(x) for x in stages],cache_scope='Qualified switch full trajectories only. Shoe holding assays provide no teacher labels. Gate repeat0 once; all attempts and diagnostic retries retained.')
    p['cache_scope']='Only qualified full switch/verified-held-shoe teacher trajectories; support assays and failed/unfinished teachers excluded. Whole-scene grouping, gate repeat0 once.'
    p['held_teacher_attempts']=held_attempts
    out=new_run('fixed_acceptance',group=('reports','protocols'));write_json(out/'protocol.json',p)
    attempts.sort(key=lambda r:r['point']);write_json(out/'batch_result.json',dict(status='complete',attempts=attempts,reused_gate=p['reuse_gate_run'],
        prior_attempts=[b.get('prior_attempts',[]) for b in batches],consolidator_sha256=digest(Path(__file__))))
    print('CONSOLIDATED',out,flush=True)

if __name__=='__main__':main()
