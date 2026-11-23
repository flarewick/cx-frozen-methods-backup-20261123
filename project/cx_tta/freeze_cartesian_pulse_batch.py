"""Freeze baseline-motion neighborhoods in original TRAIN scenes; no outcome selection."""
import json
from pathlib import Path
from .audit import digest
from .paths import ROOT,write_json
from .probe_action_replay import load_source
from .robot_kinematics import RobotFK
from .cartesian_pulse_batch import DEFAULT_PLAN,validate_plan,validate_sources,verify_run


def main():
    original_path=ROOT/'configs/intervention_batch1_20260927.json'
    coverage_path=ROOT/'outputs/reports/audits/early_interaction_coverage_20260927.json'
    original=json.loads(original_path.read_text());coverage=json.loads(coverage_path.read_text())
    scenes={(r['task'],r['seed']):r for r in original['scenes'] if r['split']=='train'}
    coverage_rows={(r['task'],r['seed']):r for r in coverage['scenes']}
    sources={(j['task'],j['seed']):j['source'] for j in original['jobs']}
    hashes=dict(json.loads((ROOT/'configs/short_window_batch1_20260927.json').read_text())['code_hashes'])
    for path in ('cx_tta/collect_cartesian_pulses.py','cx_tta/audit_cartesian_pulses.py','cx_tta/cartesian_pulse.py','cx_tta/pulse_contacts.py','cx_tta/robot_kinematics.py','cx_tta/audit_cartesian_proposals.py','cx_tta/cartesian_pulse_batch.py','cx_tta/freeze_cartesian_pulse_batch.py','scripts/collect_cartesian_pulses.sh','scripts/collect_cartesian_pulse_batch.sh'):
        hashes[path]=digest(ROOT/path)
    pilot_paths=[ROOT/'outputs/cartesian_pulses/beat_block_hammer/windows_20260928_042910_510578',ROOT/'outputs/cartesian_pulses/place_empty_cup/windows_20260928_044801_277678']
    pilots={}
    for path in pilot_paths:
        m=json.loads((path/'manifest.json').read_text());pilots[m['task'],m['source_seed'],m['chunk'],m['pulse_start']]=path
    jobs=[];external={}
    for key,scene in sorted(scenes.items()):
        row=coverage_rows[key]
        if row['split']!='train':raise ValueError('Coverage split differs')
        source=Path(sources[key]);_,_,trace=load_source(source,key[1]);full=[i for i,r in enumerate(trace) if r['executed']==32]
        anchor=row['first_motion_chunk']
        selected=sorted({i for i in (anchor-1,anchor,anchor+1) if i in full}) if anchor is not None else sorted({full[0],full[(len(full)-1)//2],full[-1]})
        if not selected:raise ValueError('No eligible neighborhood')
        config=json.loads((source/'resolved_config.json').read_text())
        for side in ('left','right'):
            urdf=RobotFK(config,side).urdf;external[str(urdf)]=digest(urdf)
        for chunk in selected:
            job=dict(task=key[0],seed=key[1],source=str(source),chunk=chunk,pulse_start=8,split='train',id=f'{key[0]}_s{key[1]}_c{chunk:03d}_p08',trace_sha256=digest(source/'traces'/f'r0_seed{key[1]}.jsonl'),source_config_sha256=digest(source/'resolved_config.json'),baseline_first_motion_chunk_offline=anchor,selection_role_offline='no_motion_temporal_stratum' if anchor is None else ('before' if chunk<anchor else 'during' if chunk==anchor else 'after'))
            pilot=pilots.get((*key,chunk,8))
            if pilot:job.update(reuse_run=str(pilot),reuse_manifest_sha256=digest(pilot/'manifest.json'))
            jobs.append(job)
    plan=dict(schema_version=1,batch_id='cartesian_train_neighborhoods_v1_20260928',record_stride=4,pulse_start=8,original_plan_sha256=digest(original_path),coverage_report_sha256=digest(coverage_path),code_hashes=hashes,external_readonly_hashes=external,jobs=jobs,scope='Original TRAIN scenes only; mechanism supplement, not blind benchmark or success-rate evaluation',selection='For each original train scene: baseline first offline motion chunk plus preceding/following full 32-action chunks. Partial terminal chunks omitted. No baseline motion: first/middle/last full chunks. Same pulse_start=8 and all 12 directions everywhere. No branch response, success gain, validation effect or task-specific direction used to select. All available phase points kept.',sampling_limitations='16 independent train scenes, not new scene diversity beyond the original frozen split. Before/during/after refer to 32-action baseline-motion neighborhoods, not verified contact phase. Some neighborhoods truncated by early baseline success. Offline selection never used as deployment router.')
    validate_plan(plan);validate_sources(jobs)
    for job in jobs:
        if job.get('reuse_run'):verify_run(Path(job['reuse_run']),job,plan)
    with DEFAULT_PLAN.open("x",encoding="utf-8") as f:
        json.dump(plan,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps(dict(plan=str(DEFAULT_PLAN),scene_count=len(scenes),checkpoints=len(jobs),reuse=sum(bool(j.get('reuse_run')) for j in jobs),branches=13*len(jobs)),indent=2))

if __name__=='__main__':main()
