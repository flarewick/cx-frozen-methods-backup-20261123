"""Freeze remaining developer points; retain gate once, never open new scenes."""
import json
from pathlib import Path
from .audit import digest
from .paths import ROOT,new_run,write_json

PARENT=ROOT/'outputs/reports/protocols/multiscene_teacher_20261003_020749_696293'

def main():
    p=json.loads((PARENT/'protocol.json').read_text());gate=json.loads((PARENT/'gate.json').read_text())
    if not gate['passed']:raise ValueError('Parent gate required')
    for file,h in {**p['source_sha256'],**p['code_sha256'],**gate['evidence_sha256']}.items():
        if digest(Path(file))!=h:raise ValueError('Changed parent '+file)
    files=[ROOT/'cx_tta'/f for f in ('prepare_fixed_teachers.py','probe_fixed_teacher.py','probe_fixed_shoe_support.py',
        'run_fixed_teachers.py','handoff_support.py','audit_local_geometry.py','robot_kinematics.py')]
    files.extend(ROOT/'scripts'/f for f in ('probe_fixed_teacher.sh','probe_fixed_shoe_support.sh'))
    p.update(version='S2d-fixed-teachers-v1',parent_protocol=str(PARENT/'protocol.json'),
        parent_gate=str(PARENT/'gate.json'),reuse_gate_run=gate['runs'][0],remaining_switch_points=list(range(5,16)),
        shoe_support_points=list(range(4)),
        selection='All 16 existing forks retained; gate point4 repeat0 reused once; 11 switch attempts and 4 shoe support assays. No new seeds.',
        shoe_support='Actual finger entity identities for both arms; unique two-finger contact owner, then independent exact-fork 12-native-action hold and 1cm lift branches. '
          'Every action: both finger contacts, relative drift <=3mm and <=3deg. Final EE goal <=5mm/3deg; lift EE rise >=8mm and object rise >=5mm. '
          'No unique owner -> support unqualified (not proof of absence). No unconditional regrasp; full shoe continuation needs separate qualified entry.',
        cache_scope='Only audited successful full teacher trajectories; failures and support assays remain diagnostics. No adapter/router/TTA training.')
    p['source_sha256'].update({str(PARENT/f):digest(PARENT/f) for f in ('protocol.json','gate.json')})
    p['source_sha256'].update(gate['evidence_sha256'])
    p['code_sha256'].update({str(f):digest(f) for f in files})
    out=new_run('fixed_teachers',group=('reports','protocols'));write_json(out/'protocol.json',p);print(out,flush=True)

if __name__=='__main__':main()
