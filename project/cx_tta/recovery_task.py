"""Run the two preselected points of one task after an independently audited gate."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from .audit import digest
from .collect_recovery import DEFAULT_PROTOCOL
from .paths import ROOT,new_run,write_json,replace_json


def reusable_runs(batch,task,points,protocol,gate_result):
    """Reuse only whole audited checkpoints, never partial branches or caches."""
    batch=batch.resolve()
    if not batch.is_relative_to(ROOT/'outputs/recovery_batches'):raise ValueError('Resume batch must belong to this project')
    manifest=json.loads((batch/'manifest.json').read_text())
    if (manifest['task']!=task or manifest['points']!=points or manifest['protocol_sha256']!=digest(protocol)
            or manifest['gate_result_sha256']!=digest(gate_result)):
        raise ValueError('Resume batch task/points/protocol/gate mismatch')
    status_file=batch/('result.json' if (batch/'result.json').exists() else 'progress.json')
    previous=json.loads(status_file.read_text());found={}
    from .audit_recovery import audit
    for row in previous.get('runs',[]):
        point=row['point'];run=Path(row['run']).resolve()
        if point not in points or point in found or not run.is_relative_to(ROOT/'outputs/recovery_continuations'):
            raise ValueError('Invalid or duplicate reused checkpoint')
        saved=json.loads((run/'manifest.json').read_text())
        if saved['phase']!='pilot' or saved['point_index']!=point or saved['task']!=task or saved['protocol_sha256']!=digest(protocol):
            raise ValueError('Reused checkpoint identity mismatch')
        if digest(run/'result.json')!=row['result_sha256'] or digest(run/'audit.json')!=row['audit_sha256']:
            raise ValueError('Reused result/audit changed')
        # Recheck the actual data and current runner/assets, not just the old status.
        checked=audit(run);old=json.loads((run/'audit.json').read_text())
        if checked!=old:raise ValueError('Reused checkpoint no longer matches its audit')
        found[point]=dict(row,reused_from_batch=str(batch))
    if not found:raise ValueError('Resume batch has no complete audited checkpoint')
    return found


def collect_point(command,batch,point):
    run=None
    with (batch/f'point_{point}.log').open('x') as log:
        with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
            try:
                for line in child.stdout:
                    log.write(line);log.flush();sys.stdout.write(line);sys.stdout.flush()
                    if line.startswith('RUN '):
                        if run is not None:raise ValueError('Duplicate run announcement')
                        run=Path(line[4:].strip()).resolve()
                        if not run.is_relative_to(ROOT/'outputs/recovery_continuations'):raise ValueError('Unexpected child output')
                code=child.wait()
            except BaseException:
                child.terminate();child.wait();raise
        if code:raise subprocess.CalledProcessError(code,command)
    if run is None:raise ValueError('No run announcement')
    from .audit_recovery import audit
    checked=audit(run);write_json(run/'audit.json',checked)
    return dict(point=point,run=str(run),audit_sha256=digest(run/'audit.json'),result_sha256=checked['result_sha256'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--task',required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--gate-result',required=True,type=Path)
    p.add_argument('--protocol',default=DEFAULT_PROTOCOL,type=Path);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume-batch',type=Path,help='Reuse complete audited points from this task batch; rerun only missing points')
    a=p.parse_args()
    a.gate_result=a.gate_result.resolve();a.protocol=a.protocol.resolve()
    gate=json.loads(a.gate_result.read_text());audit=json.loads((a.gate_result.parent/'audit.json').read_text())
    if gate['status']!='complete' or not gate['gate_passed'] or audit['status']!='passed' or audit['result_sha256']!=digest(a.gate_result) or audit['manifest_sha256']!=digest(a.gate_result.parent/'manifest.json'):
        raise ValueError('Independent gate audit missing, invalid or stale')
    protocol=json.loads(a.protocol.read_text());points=[i for i,r in enumerate(protocol['points']) if r['task']==a.task]
    if len(points)!=2:raise ValueError('Require exactly two preselected task points')
    reused=reusable_runs(a.resume_batch,a.task,points,a.protocol,a.gate_result) if a.resume_batch else {}
    for point,row in reused.items():print('REUSE_POINT',point,row['run'],flush=True)
    pending=[point for point in points if point not in reused]
    commands=[['bash',str(ROOT/'scripts/collect_recovery.sh'),'--phase','pilot','--point',str(point),'--protocol',str(a.protocol),
        '--gate-result',str(a.gate_result),'--gpu',a.gpu] for point in pending]
    if a.dry_run:
        for command in commands:subprocess.run(command+['--dry-run'],cwd=ROOT,check=True)
        return
    batch=new_run('pilot',group=('recovery_batches',a.task));print('BATCH '+str(batch),flush=True)
    write_json(batch/'manifest.json',dict(task=a.task,points=points,commands=commands,protocol=str(a.protocol),
        pending_points=pending,resume_batch=str(a.resume_batch.resolve()) if a.resume_batch else None,
        reused_runs=list(reused.values()),
        protocol_sha256=digest(a.protocol),gate_result=str(a.gate_result),gate_result_sha256=digest(a.gate_result),
        wrapper_sha256=digest(Path(__file__)),audit_code_sha256=digest(ROOT/'cx_tta/audit_recovery.py')))
    runs=list(reused.values())
    try:
        replace_json(batch/'progress.json',dict(status='running',runs=runs))
        for point,command in zip(pending,commands):
            runs.append(collect_point(command,batch,point))
            replace_json(batch/'progress.json',dict(status='running',runs=runs))
        runs.sort(key=lambda row:row['point'])
        results=[json.loads((Path(r['run'])/'result.json').read_text()) for r in runs]
        rows=[row for r in results for row in r['reports']]
        summary={name:dict(success=sum(r['success'] for r in rows if r['branch']==name),
            total=sum(r['branch']==name for r in rows)) for name in ('base','sham','plus','minus')}
        pairs=[pair for r in results for pair in r['pairs']]
        write_json(batch/'result.json',dict(status='complete',runs=runs,summary=summary,pairs=pairs,
            scope='Two training-scene checkpoints with correlated repeated continuations, not benchmark task success rate.'))
        replace_json(batch/'progress.json',dict(status='complete',runs=runs))
        for name,counts in summary.items():
            print(f"{a.task} {name}: {counts['success']}/{counts['total']} ({counts['success']/counts['total']:.1%}); continuation diagnostic",flush=True)
        print('BATCH_COMPLETE '+str(batch),flush=True)
    except BaseException:
        replace_json(batch/'result.json',dict(status='failed_or_interrupted',runs=runs,traceback=traceback.format_exc(),
            scope='Incomplete branches are censored, never task failures. Child logs and partial artifacts retained.'))
        raise


if __name__=='__main__':main()
