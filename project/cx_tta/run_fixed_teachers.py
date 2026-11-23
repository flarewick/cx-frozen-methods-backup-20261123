"""Bounded sequential fresh-process attempts, durable results including failures."""
import argparse,json,subprocess,time
from pathlib import Path
from .paths import ROOT,write_json,replace_json
from .audit import digest

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    p=json.loads(a.protocol.read_text());folder=a.protocol.parent;records=[]
    if not a.gpu.isdigit():raise ValueError('GPU index')
    jobs=[('shoe_support',i) for i in p['shoe_support_points']]+[('switch',i) for i in p['remaining_switch_points']]
    # Refuse silent retries/overwriting failures. A changed protocol is required to retry.
    if any((folder/f'{kind}_{i}.log').exists() for kind,i in jobs):raise ValueError('Attempt already exists')
    for kind,i in jobs:
        replace_json(folder/'batch_progress.json',dict(phase='running',kind=kind,point=i,completed=records))
        script='probe_fixed_shoe_support.sh' if kind=='shoe_support' else 'probe_fixed_teacher.sh'
        started=time.monotonic();log=folder/f'{kind}_{i}.log'
        with log.open('x') as f:
            result=subprocess.run(['bash',str(ROOT/'scripts'/script),'--protocol',str(a.protocol.resolve()),'--point',str(i),'--gpu',a.gpu],stdout=f,stderr=subprocess.STDOUT,timeout=1020)
        paths=[line.strip() for line in log.read_text().splitlines() if line.startswith(str(ROOT/'outputs')+'/')]
        record=dict(kind=kind,point=i,exit_code=result.returncode,seconds=time.monotonic()-started,
                    run=paths[0] if paths else None,log=str(log),log_sha256=digest(log))
        write_json(folder/f'{kind}_{i}_attempt.json',record);records.append(record);print('ATTEMPT',json.dumps(record),flush=True)
        if result.returncode:
            # Stop interface failures for diagnosis; task-level negative outcomes return zero.
            replace_json(folder/'batch_progress.json',dict(phase='interface_failure',completed=records));raise SystemExit(result.returncode)
    write_json(folder/'batch_result.json',dict(status='complete',attempts=records,reused_gate=p['reuse_gate_run']))
    replace_json(folder/'batch_progress.json',dict(phase='complete',completed=records))

if __name__=='__main__':main()
