"""Copy the audited inventory into the Git and evidence trees."""
from pathlib import Path
import json
import shutil
from collections import Counter
import posixpath
import sys

from collect import Collector, GATE, relpath, sha

ROOT=Path(__file__).resolve().parents[1]
source=Path(sys.argv[1]).resolve()
preview=ROOT/'inventory-preview.json'
plan=json.loads(preview.read_text())
assert Path(plan['source_project'])==source
c=Collector(source,ROOT)
selected=[]
omitted=[]
for file in plan['files']:
    if file['path'].startswith('multi_event_gate_v1/outputs/task_adapters/shared6_2642e508106d2049/collections/') and file['path'].endswith('.hdf5'):
        if not any('/study.json' in r or 'frozen_tta093_v1/manifest.json' in r for r in file['reasons']):
            omitted.append(file)
            continue
    selected.append(file)
plan['files']=selected
plan['training_data_provenance_only']=omitted
for row in plan['external']:
    normalized=posixpath.normpath(row['path'])
    if normalized!=row['path'] and relpath(normalized,source) is not None:
        c.add(normalized,row['referenced_by'],row['sha256'])
    else:
        c.external.add((normalized,row['sha256'],row['referenced_by']))
gate=c.read(GATE/'manifest.json')
snapshots={}
for row in gate['files']:
    snapshots[(row['project_relative_path'],row['sha256'])]=source/GATE/row['snapshot']
for row in gate['reference_registries']:
    original=Path(row['source']).relative_to(source).as_posix()
    snapshots[(original,row['sha256'])]=source/GATE/row['snapshot']
for path in (
    'deployment/run.py','deployment/environment.json','deployment/task_adapter_python.sh',
    'deployment/reports/target_candidate_eval_prepared_cpu_20261025.json',
    'multi_event_gate_v1/scripts/eval_event_full_worker.sh','multi_event_gate_v1/scripts/eval_event_full.sh',
    'multi_event_gate_v1/task_adapter_v1/scripts/eval.sh','multi_event_gate_v1/task_adapter_v1/scripts/launch.sh',
    'multi_event_gate_v1/docs/frozen_event_evaluation.md',
    'multi_event_gate_v1/cx_tta/evaluate_frozen_event.py','multi_event_gate_v1/cx_tta/frozen_event_support.py',
    'multi_event_gate_v1/cx_tta/frozen_event_policy.py','multi_event_gate_v1/cx_tta/multi_event_policy.py',
    'multi_event_gate_v1/cx_tta/stacking_policy.py',
):
    c.add(path,'显式部署命令和 source-bound CPU 预检')
legacy=Path('outputs/effect_learning/output_support_20261005_082307_793921')
for name in ('A_bounded_clipped_seed0.pt','training_protocol.json','training_result.json','training_audit.json'):
    c.add(legacy/name,'原 event_full legacy4 权重及运行时验收')
legacy_protocol=c.read(legacy/'training_protocol.json')
for field in ('design','initial','inputs','labels','development_inputs','development_labels','projection','robot_config'):
    c.add(legacy_protocol[field],'原 event_full legacy4 运行时读取: '+field)
base=relpath(legacy_protocol['parent_run'],source)
for name in ('S0_seed0.pt','training_protocol.json','training_audit.json'):
    c.add(base/name,'原 event_full legacy4 基线合同')
ref=Path('outputs/reports/protocols/shared_onset_expansion_20261007_145526_085093/protocol.json')
c.add(ref,'原 event_full 官方参考任务与运行时环境')
for row in c.read(ref)['reference'].values():
    folder=relpath(row['run'],source)
    c.add(folder/'provenance.json','原 event_full 审计来源')
    for name in ('envs','task_config','description'):
        c.directory(folder/'robotwin'/name,'原 event_full 实际复制的原生代码和配置')
    c.external.add((str(folder/'robotwin/assets'),None,'只读官方 RoboTwin assets 挂载'))
c.follow()
old={row['path']:row for row in plan['files']}
todo={p for p in c.files if str(p) not in old}
c.files={p:c.files[p] for p in todo}
additional,issues=c.inventory()
for path,(_,digest,size) in additional.items():
    q=str(path)
    old[q]=dict(path=q,sha256=digest,bytes=size,reasons=sorted(c.files[path]),
                layer='evidence' if c.evidence(path,size) else 'git')
plan['files']=sorted(old.values(),key=lambda r:r['path'])
plan['unresolved_bindings'].extend(issues)
plan['external']=[dict(path=p,sha256=h,referenced_by=r)for p,h,r in sorted(c.external,key=str)]
assert not any(row['issue']=='credential-pattern or private path excluded' for row in plan['unresolved_bindings'])
for number,row in enumerate(plan['files'],1):
    q=Path(row['path'])
    src=source/q
    if not src.is_file() or sha(src)!=row['sha256']:
        src=snapshots[(str(q),row['sha256'])]
    top=ROOT/('project' if row['layer']=='git' else '_evidence_staging')
    dest=top/q
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
    assert sha(dest)==row['sha256'],str(q)
    if number % 1000 == 0:
        print(f'Copied and verified {number}/{len(plan["files"])}',flush=True)
plan['summary']={'files':len(plan['files']),'bytes':sum(r['bytes']for r in plan['files']),
                 'by_layer':dict(Counter(r['layer']for r in plan['files'])),
                 'issues':dict(Counter(r['issue']for r in plan['unresolved_bindings'])),'external':len(plan['external']),
                 'offline_training_data_references':len(omitted)}
with (ROOT/'inventory.json').open('x') as dest:
    json.dump(plan,dest,ensure_ascii=False,indent=2)
    dest.write('\n')
print(json.dumps(plan['summary']),flush=True)
