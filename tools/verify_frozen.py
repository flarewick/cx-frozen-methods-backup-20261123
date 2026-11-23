"""Compare selected original freeze digests against the restored files."""
import argparse
from collections import Counter
import json
from pathlib import Path
from snapshot import sha

W=Path('multi_event_gate_v1')
T=W/'tta_event_v1'
GATE=W/'frozen/all_candidate_gate_v1_20261101'
MANIFESTS={
    'event_full_400': Path('deployment/reports/target_cross_task_shared250_a050_full400_audit_20261025.json'),
    'TTA093': T/'frozen_tta093_v1/manifest.json',
    'TTA78_plain': T/'outputs/native_seven_official_v9/plain_001/study.json',
    'TTA78_gated': T/'outputs/native_seven_official_v9/gated_000/study.json',
    'TTA78_fit_qualification': T/'outputs/native_fit7_postexit_recovery_v2/qualification.json',
}
KEYS=('source_sha256','input_sha256','reference_input_sha256','frozen_files_sha256')
ROOT=Path(__file__).resolve().parents[1]


def verify(project):
    idx=json.loads((ROOT/'inventory.json').read_text())
    aliases=idx['historical_versions_in_frozen_snapshot']
    prefixes=idx['original_project_aliases']
    counts={}
    unverified=[]
    memo={}
    def digest(path):
        if path not in memo:
            memo[path]=sha(path) if path.is_file() else None
        return memo[path]
    def check(label,name,wanted):
        rel=name
        for prefix in prefixes:
            if name.startswith(prefix):
                rel=name.removeprefix(prefix)
                break
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts:
            unverified.append(dict(method=label,path=name,reason='external'))
            return
        actual=digest(project/path)
        if actual!=wanted and (alt:=aliases.get(rel+'@'+wanted)):
            actual=digest(project/alt)
        if actual!=wanted:
            unverified.append(dict(method=label,path=rel,expected=wanted,actual=actual))
    for label,rel in MANIFESTS.items():
        d=json.loads((project/rel).read_text())
        entries=[(name,wanted)for k in KEYS for name,wanted in d.get(k,{}).items()]
        for name,wanted in entries:
            check(label,name,wanted)
        counts[label]=len(entries)
    manifest=json.loads((project/GATE/'manifest.json').read_text())
    entries=[(str(GATE/row['snapshot']),row['sha256'])for k in ('files','reference_registries')for row in manifest[k]]
    for name,wanted in entries:
        check('all_event_v118',name,wanted)
    counts['all_event_v118']=len(entries)
    per_method=Counter(row['method']for row in unverified)
    print(json.dumps(dict(bindings=counts,unverified_by_method=dict(per_method),unverified=unverified),ensure_ascii=False,indent=2))
    return bool(unverified)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('restored_project',type=Path)
    raise SystemExit(verify(p.parse_args().restored_project.resolve()))
