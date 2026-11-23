"""Finish bounded pilot checks once the existing collector exits; no GPU work."""
import argparse,json,time
from pathlib import Path
from .paths import ROOT,new_run,output_path,write_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    args=parser.parse_args();runs=[output_path(p) for p in args.runs]
    status_dir=new_run('pulse_finalization',group=('reports','audits'))
    print('status_directory='+str(status_dir),flush=True)
    start=time.monotonic();process=Path('/proc')/str(args.wait_pid)/'cmdline'
    while process.exists():
        try:cmdline=process.read_bytes()
        except FileNotFoundError:break
        if b'cx_tta.collect_cartesian_pulses' not in cmdline:break
        if time.monotonic()-start>3600:raise TimeoutError('Pilot still running after bounded wait; collector left untouched')
        time.sleep(10)
    try:
        from .analyze_cartesian_pulses import analyze
        from .cartesian_pulse_dataset import build
        from .audit_pulse_dataset import audit
        analyses=[]
        for run in runs:
            if (run/'failure.json').exists():raise RuntimeError('Collector failed: '+str(run))
            saved_audit=json.loads((run/'schema_audit.json').read_text())
            if saved_audit['status']!='passed':raise RuntimeError('Collector audit not passed: '+str(run))
            report=analyze(run)
            target=status_dir/(report['task']+'_mechanism.json')
            write_json(target,report);analyses.append(dict(task=report['task'],run=str(run),analysis=str(target),summary=report['summary']))
            print('AUDIT_ANALYSIS_PASS '+report['task'],flush=True)
        dataset=build(runs,ROOT/'configs/intervention_batch1_20260927.json')
        verification=audit(dataset);write_json(dataset/'interface_audit.json',verification)
        result=dict(status='complete',analyses=analyses,dataset=str(dataset),interface_audit=verification,scope='Bounded collection/learning-interface verification only. No training, new collection, validation tuning, success-rate or transfer claim.')
        write_json(status_dir/'result.json',result)
        with (ROOT/'docs/plan.md').open('a') as f:
            f.write('\n## 两个schema4门检自动后处理完成\n\n两个任务采集审计与离线作用分析通过；共同起点数据接口审计通过。报告：'+str(status_dir.relative_to(ROOT))+'/result.json；数据接口：'+str(dataset.relative_to(ROOT))+'。这里只包含'+str(verification['independent_scenes'])+'个独立场景，未启动训练或正式补采，不能声称跨任务泛化。下一步读取完整作用/接触/时域结果，再冻结覆盖补集。\n')
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        import traceback
        write_json(status_dir/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
