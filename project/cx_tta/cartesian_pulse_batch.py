"""Frozen train-scene Cartesian mechanism batch; separate task jobs and resumption."""
import argparse
from datetime import datetime,timezone
import fcntl,json,subprocess
from pathlib import Path
from .audit import digest
from .audit_cartesian_pulses import audit_run
from .paths import ROOT,output_path,write_json,replace_json
from .probe_action_replay import load_source,source_runtime

DEFAULT_PLAN=ROOT/'configs/cartesian_pulse_batch1_20260928.json'


def validate_plan(plan):
    if plan['schema_version']!=1 or not plan['batch_id'].isidentifier() or plan['record_stride']!=4 or plan['pulse_start']!=8:
        raise ValueError('Unsupported frozen plan')
    for path,sha in plan['code_hashes'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Frozen code changed: '+path)
    for path,sha in plan['external_readonly_hashes'].items():
        if digest(Path(path))!=sha:raise ValueError('Robot description changed: '+path)
    original_path=ROOT/'configs/intervention_batch1_20260927.json'
    if digest(original_path)!=plan['original_plan_sha256']:raise ValueError('Original split changed')
    coverage_path=ROOT/'outputs/reports/audits/early_interaction_coverage_20260927.json'
    if digest(coverage_path)!=plan['coverage_report_sha256']:raise ValueError('Selection report changed')
    original=json.loads(original_path.read_text());groups={(r['task'],r['seed']) for r in original['scenes'] if r['split']=='train'}
    seen=set();covered=set()
    for j in plan['jobs']:
        identity=(j['task'],j['seed'],j['chunk'],j['pulse_start'])
        if identity in seen or j['split']!='train' or (j['task'],j['seed']) not in groups or j['pulse_start']!=8 or not j['id'].isidentifier():
            raise ValueError('Duplicate job, split leakage or changed intervention')
        seen.add(identity);covered.add((j['task'],j['seed']))
    if covered!=groups:raise ValueError('Missing frozen train scene')
    if len({j['id'] for j in plan['jobs']})!=len(plan['jobs']):raise ValueError('Repeated job ID')
    return True


def validate_sources(jobs):
    cached={}
    for j in jobs:
        source=Path(j['source']);key=(j['source'],j['seed'])
        if key not in cached:
            source_runtime(source);cached[key]=load_source(source,j['seed'])
        manifest,_,rows=cached[key]
        if manifest['args']['task']!=j['task'] or not 0<=j['chunk']<len(rows) or rows[j['chunk']]['executed']!=32:
            raise ValueError('Source task or full chunk differs')
        if digest(source/'traces'/f"r0_seed{j['seed']}.jsonl")!=j['trace_sha256'] or digest(source/'resolved_config.json')!=j['source_config_sha256']:
            raise ValueError('Frozen source changed')


def verify_run(run,job,plan):
    run=output_path(run);m=json.loads((run/'manifest.json').read_text())
    expected=dict(source=job['source'],source_seed=job['seed'],task=job['task'],chunk=job['chunk'],pulse_start=job['pulse_start'],schema_version=4,record_stride=4,proposal_set='cartesian_pulse_v1')
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Attempt identity differs: '+str(run))
    hashes={'collector_sha256':'cx_tta/collect_cartesian_pulses.py','proposal_code_sha256':'cx_tta/cartesian_pulse.py','fk_code_sha256':'cx_tta/robot_kinematics.py','solver_code_sha256':'cx_tta/audit_cartesian_proposals.py','contact_code_sha256':'cx_tta/pulse_contacts.py','contract_sha256':'cx_tta/intervention_contract.py','window_contract_sha256':'cx_tta/short_window_contract.py'}
    if any(m.get(k)!=plan['code_hashes'][path] for k,path in hashes.items()):raise ValueError('Attempt code provenance differs')
    return audit_run(run)


def completed_attempt(attempts,job,plan):
    if not attempts.exists():return None
    for run in sorted(attempts.iterdir()):
        if not run.is_dir() or not (run/'result.json').exists() or (run/'failure.json').exists():continue
        verify_run(run,job,plan);return run
    return None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,default=DEFAULT_PLAN);p.add_argument('--task',required=True,choices=('beat_block_hammer','place_shoe','turn_switch','place_empty_cup'));p.add_argument('--gpu',default='0');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not a.gpu.isdigit():p.error('Invalid GPU')
    plan=json.loads(a.plan.read_text());validate_plan(plan);jobs=[j for j in plan['jobs'] if j['task']==a.task];validate_sources(jobs)
    if not jobs:p.error('No jobs for task')
    for j in jobs:
        if j.get('reuse_run'):
            if digest(Path(j['reuse_run'])/'manifest.json')!=j['reuse_manifest_sha256']:raise ValueError('Reuse manifest changed')
            verify_run(Path(j['reuse_run']),j,plan)
    print(json.dumps(dict(task=a.task,scenes=len({j['seed'] for j in jobs}),checkpoints=len(jobs),branches=13*len(jobs),reuse_checkpoints=sum(bool(j.get('reuse_run')) for j in jobs),new_checkpoints=sum(not bool(j.get('reuse_run')) for j in jobs),dry_run=a.dry_run)),flush=True)
    if a.dry_run:return
    dest=output_path(ROOT/'outputs/cartesian_pulse_batches'/plan['batch_id']/a.task);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Task batch already running')
        if (dest/'plan.json').exists():
            if json.loads((dest/'plan.json').read_text())!=plan:raise ValueError('Existing batch plan differs')
        else:write_json(dest/'plan.json',plan)
        records=[]
        for j in jobs:
            attempts=dest/'jobs'/j['id']/'attempts'
            run=Path(j['reuse_run']) if j.get('reuse_run') else completed_attempt(attempts,j,plan)
            reused=run is not None
            if run is None:
                attempts.mkdir(parents=True,exist_ok=True);stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');run=attempts/('attempt_'+stamp);log=attempts/('attempt_'+stamp+'.log')
                command=['bash',str(ROOT/'scripts/collect_cartesian_pulses.sh'),'--source',j['source'],'--seed',str(j['seed']),'--chunk',str(j['chunk']),'--pulse-start',str(j['pulse_start']),'--gpu',a.gpu,'--run-dir',str(run)]
                print(f"START {j['id']} log={log}",flush=True)
                with log.open('x') as stream:status=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
                if status:
                    replace_json(dest/'progress.json',dict(status='failed',completed=records,failed_job=j,returncode=status,log=str(log)))
                    raise RuntimeError('Collection failed; same command resumes. See '+str(log))
            verify_run(run,j,plan)
            records.append(dict(j,run=str(run),reused=reused,manifest_sha256=digest(run/'manifest.json'),result_sha256=digest(run/'result.json')))
            replace_json(dest/'progress.json',dict(status='running',completed=records,total_jobs=len(jobs)))
            print(f"PASS {j['id']} ({len(records)}/{len(jobs)}), reused={reused}",flush=True)
        replace_json(dest/'index.json',dict(status='complete',task=a.task,plan_sha256=digest(a.plan),records=records,scope='TRAIN-only mechanism supplement, not success-rate evaluation'))
        replace_json(dest/'progress.json',dict(status='complete',completed=records,total_jobs=len(jobs)))
        print(f"COMPLETE {a.task}: {len(records)}/{len(jobs)} points, {13*len(records)} branches; index={dest/'index.json'}",flush=True)

if __name__=='__main__':main()
