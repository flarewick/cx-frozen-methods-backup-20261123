"""Resume frozen acceptance after a documented diagnostic-alias repair."""
import argparse,json,subprocess,time
from pathlib import Path
from .audit import digest
from .paths import ROOT,new_run,write_json,replace_json

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--parent',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if not a.gpu.isdigit():raise ValueError('GPU index')
    if a.parent:
        parent=a.parent.resolve();p=json.loads((parent/'protocol.json').read_text());progress=json.loads((parent/'batch_progress.json').read_text())
        if progress['phase']=='running':raise ValueError('Parent still running')
        completed=progress['completed'];reused=[];jobs=[]
        for kind,points in [('shoe_support',p['shoe_support_points']),('switch',p['remaining_switch_points'])]:
            for point in points:
                prior=next((r for r in completed if r['kind']==kind and r['point']==point),None)
                if prior and prior['exit_code']==0:
                    reused.append(prior);continue
                if prior:
                    path=Path(prior['run']);failure=json.loads((path/'failure.json').read_text())
                    if not (path/'replay_mismatch.json').exists() and 'Teacher fork differs from original' not in failure['traceback']:
                        raise ValueError('Repair does not address failure '+str(path))
                jobs.append(dict(kind=kind,point=point,previous_attempt=prior))
        files=[parent/'protocol.json']
        for old in completed:
            files.extend([Path(old['log']),Path(old['run'])/'manifest.json'])
            files.extend(Path(old['run'])/name for name in ('result.json','qualification.json','failure.json','replay_mismatch.json') if (Path(old['run'])/name).exists())
        code=[ROOT/'cx_tta'/x for x in ('complete_fixed_acceptance.py','replay_active_objects.py','probe_fixed_shoe_support_v3.py','probe_fixed_teacher_v3.py')]
        code.extend(ROOT/'scripts'/x for x in ('probe_fixed_shoe_support_v3.sh','probe_fixed_teacher_v3.sh'))
        p.update(version='S2d-active-entity-acceptance-v3',previous_protocol=str(parent/'protocol.json'),
            jobs=jobs,reused_attempts=reused,all_prior_attempts=completed,
            amendment='Exclude only proven stale cross-episode cluttered_obj diagnostic alias, retaining raw comparison and all failed attempts. Exact RGB/command and 1e-5 physical tolerance unchanged. No new seeds or repeat of an executed teacher outcome.')
        p['source_sha256'].update({str(f.resolve()):digest(f) for f in files});p['code_sha256'].update({str(f):digest(f) for f in code})
        out=new_run('fixed_acceptance',group=('reports','protocols'));write_json(out/'protocol.json',p);print(out,flush=True);return
    if not a.run:ap.error('--parent to prepare or --run to execute')
    folder=a.run.resolve();p=json.loads((folder/'protocol.json').read_text());records=list(p['reused_attempts'])
    for job in p['jobs']:
        kind,i=job['kind'],job['point'];log=folder/f'{kind}_{i}.log'
        if log.exists():raise ValueError('No silent retry')
        replace_json(folder/'batch_progress.json',dict(phase='running',kind=kind,point=i,completed=records))
        script='probe_fixed_shoe_support_v3.sh' if kind=='shoe_support' else 'probe_fixed_teacher_v3.sh';start=time.monotonic()
        with log.open('x') as f:
            status=subprocess.run(['bash',str(ROOT/'scripts'/script),'--protocol',str(folder/'protocol.json'),'--point',str(i),'--gpu',a.gpu],stdout=f,stderr=subprocess.STDOUT,timeout=1020)
        paths=[line.strip() for line in log.read_text().splitlines() if line.startswith(str(ROOT/'outputs')+'/')]
        record=dict(kind=kind,point=i,exit_code=status.returncode,seconds=time.monotonic()-start,
                    run=paths[0] if paths else None,log=str(log),log_sha256=digest(log),previous_attempt=job['previous_attempt'])
        write_json(folder/f'{kind}_{i}_attempt.json',record);records.append(record);print('ATTEMPT',json.dumps(record),flush=True)
        if status.returncode and not (record['run'] and (Path(record['run'])/'replay_mismatch.json').exists()):
            replace_json(folder/'batch_progress.json',dict(phase='interface_failure',completed=records));raise SystemExit(status.returncode)
    records.sort(key=lambda r:r['point'])
    write_json(folder/'batch_result.json',dict(status='complete',attempts=records,reused_gate=p['reuse_gate_run'],prior_attempts=p['all_prior_attempts']))
    replace_json(folder/'batch_progress.json',dict(phase='complete',completed=records))

if __name__=='__main__':main()
