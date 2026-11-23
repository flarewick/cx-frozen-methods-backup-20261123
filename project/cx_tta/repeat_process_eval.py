"""Two independent evaluation processes for one task/condition; 4 episodes total."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from .paths import ROOT,new_run,write_json,replace_json
from .evaluate_b2 import TASKS,read_legacy_l0


def command(task,condition,gpu,mode,legacy):
    return ['bash',str(ROOT/'scripts/eval_b2.sh'),'--task',task,'--condition',condition,
            '--legacy-l0',str(legacy),'--seeds','100380,100381','--repeats','1','--trace',
            '--inference-mode',mode,'--policy-init-seed','0','--gpu',gpu]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--condition',choices=('L0','B2'),required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--inference-mode',choices=('compiled','eager'),required=True)
    p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if not args.gpu.isdigit():p.error('GPU must be nonnegative integer')
    suffix='_20260902_181740' if args.task=='place_shoe' else ''
    legacy=ROOT.parent/f'direction_guide_TTA_v5/outputs/scripted_u/attr_L0_{args.task}_demo_randomized_100380_100399{suffix}/episodes.jsonl'
    rows=read_legacy_l0(legacy,args.task)
    if any(s not in rows for s in (100380,100381)):p.error('Frozen diagnostic seeds missing')
    cmd=command(args.task,args.condition,args.gpu,args.inference_mode,legacy)
    if args.dry_run:
        print(shlex.join(cmd),flush=True)
        subprocess.run(cmd+['--dry-run'],check=True,cwd=ROOT)
        return
    run=new_run('process_pair',group=('diagnostics',args.task,args.condition));print(run,flush=True)
    write_json(run/'manifest.json',{'scope':'process variance diagnostic, two scenes repeated across two processes; not four independent scenes',
                                  'command':cmd,'processes':2,'seeds':[100380,100381]})
    records=[]
    try:
        for repeat in range(2):
            eval_run=None
            with (run/f'process_{repeat}.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                try:
                    for line in child.stdout:
                        log.write(line);log.flush();print(line,end='',flush=True)
                        candidate=line.strip()
                        if candidate.startswith(str(ROOT/'outputs/diagnostics')+'/') and Path(candidate).is_dir():
                            eval_run=Path(candidate)
                    code=child.wait()
                finally:
                    if child.poll() is None:child.terminate();child.wait()
            if code or eval_run is None:raise RuntimeError(f'process {repeat} failed: exit={code}, run={eval_run}')
            result=json.loads((eval_run/'result.json').read_text())
            if result['status']!='complete':raise RuntimeError('Incomplete child evaluation')
            records.append({'process_repeat':repeat,'run':str(eval_run),'result':result})
            replace_json(run/'processes.json',records)
        write_json(run/'result.json',{'status':'complete','processes':records,
            'unique_scenes':2,'successes':sum(r['result']['successes'] for r in records),
            'eligible_episodes':sum(r['result']['eligible_episodes'] for r in records)})
        print(f'Independent-process diagnostic complete: {run}',flush=True)
    except BaseException:
        import traceback
        write_json(run/'failure.json',{'traceback':traceback.format_exc(),'completed_processes':len(records)})
        raise
if __name__=='__main__':main()
