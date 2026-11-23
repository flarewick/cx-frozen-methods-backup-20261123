"""Complete every support-qualified fixed shoe fork before assigning labels."""
import argparse,json,subprocess,time
from pathlib import Path
from .paths import ROOT,new_run,write_json,replace_json
from .audit import digest
from .cache_recovery import read,verify_hashes
from .audit_fixed_support import audit as support_audit

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--support-stage',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if not a.gpu.isdigit():raise ValueError('Numeric GPU required')
    if a.support_stage:
        folder=a.support_stage.resolve();p=read(folder/'protocol.json');b=read(folder/'batch_result.json')
        if b['status']!='complete' or len(b['attempts'])!=4:raise ValueError('All four support points first')
        held={};inventory=[]
        for r in b['attempts']:
            path=Path(r['run'])
            if r['exit_code']:raise ValueError('Unresolved support interface')
            if not (path/'audit.json').exists():support_audit(path)
            q=read(path/'audit.json');result=read(path/'result.json')
            if not q['passed']:raise ValueError('Support audit required')
            inventory.append(dict(point=r['point'],support=str(path),held_qualified=q['held_qualified'],owner=result['owner']))
            if q['held_qualified']:held[str(r['point'])]=str(path)
            for f in [path/'manifest.json',path/'audit.json',path/'result.json',path/'hold/fork.json',*path.glob('*/*replay_check.json')]:p['source_sha256'][str(f)]=digest(f)
            p['source_sha256'].update(q['source_sha256'])
        p.update(version='S2d-fixed-held-completion-v1',held_sources=held,support_inventory=inventory,
            selection='Every support-qualified fork among the same four fixed shoe points. Fresh exact original fork, original budget, teacher seed930000, verified owner. Skip repeated pregrasp only; retain closure/lift/preplace/place/release. One attempt each.',
            qualification='Exact support fork, same owner as original teacher, audited actual native commands, successful original predicate only during release after closure/lift/preplace/place plans. No labels from failed or unfinished stages.')
        for f in (folder/'protocol.json',folder/'batch_result.json'):p['source_sha256'][str(f)]=digest(f)
        for f in [Path(__file__),ROOT/'cx_tta/probe_fixed_held_teacher.py',ROOT/'scripts/probe_fixed_held_teacher.sh',ROOT/'cx_tta/held_teacher.py']:p['code_sha256'][str(f)]=digest(f)
        out=new_run('fixed_held_teachers',group=('reports','protocols'));write_json(out/'protocol.json',p);print('PREPARED',out,flush=True);return
    if not a.run:ap.error('--support-stage or --run required')
    folder=a.run.resolve();p=read(folder/'protocol.json');verify_hashes({**p['source_sha256'],**p['code_sha256']});records=[]
    for point in p['held_sources']:
        i=int(point);log=folder/f'point_{i}.log';start=time.monotonic()
        replace_json(folder/'progress.json',dict(phase='running',point=i,completed=records))
        with log.open('x') as f:
            status=subprocess.run(['bash',str(ROOT/'scripts/probe_fixed_held_teacher.sh'),'--protocol',str(folder/'protocol.json'),'--point',point,'--gpu',a.gpu],stdout=f,stderr=subprocess.STDOUT,timeout=1020)
        paths=[line.strip() for line in log.read_text().splitlines() if line.startswith(str(ROOT/'outputs')+'/')]
        record=dict(kind='shoe_teacher',point=i,run=paths[0] if paths else None,exit_code=status.returncode,log=str(log),log_sha256=digest(log),seconds=time.monotonic()-start)
        write_json(folder/f'point_{i}_attempt.json',record);records.append(record);print('HELD_ATTEMPT',json.dumps(record),flush=True)
        if status.returncode:
            replace_json(folder/'progress.json',dict(phase='interface_failure',completed=records));raise SystemExit(status.returncode)
    write_json(folder/'result.json',dict(status='complete',attempts=records,support_inventory=p['support_inventory']))
    replace_json(folder/'progress.json',dict(phase='complete',completed=records))

if __name__=='__main__':main()
