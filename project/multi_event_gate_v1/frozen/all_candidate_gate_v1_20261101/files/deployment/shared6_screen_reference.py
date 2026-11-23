"""Compare fresh expert screens with retained, source-matched real screens.

No reference row is used for admission or to skip original planning; no policy
outcome or incomplete trajectory is imported into the new evaluation identity.
"""
import copy,hashlib,json
from pathlib import Path
FIELDS=('seed','episode_index','eligible','initial_physical_sha256','expert_plan_success','expert_success','scene_info','reason','error_type','error')
def comparable(manifest):return {k:v for k,v in manifest.items()if k not in('evaluation_identity','deployment_shared6','condition','event_config')}
def load_references(plan):
    out={};records=[]
    for job in plan['jobs']:
        run=Path(job['run']);target=comparable(job['manifest'])
        for manifest in sorted(run.parent.glob('eval_*/manifest.json')):
            previous=json.loads(manifest.read_text());queue=manifest.parent/'concurrency_queue.json'
            if manifest.parent==run or comparable(previous)!=target or not queue.is_file():continue
            value=json.loads(queue.read_text())
            if value['identity']!=previous['evaluation_identity']:raise RuntimeError('Reference queue identity differs')
            for row in value['attempts'][:3]:
                key=str(row['seed']);expected={k:copy.deepcopy(row.get(k))for k in FIELDS}
                if key in out and out[key]!=expected:raise RuntimeError('Source-matched original expert references conflict')
                out[key]=expected
            records.append(dict(manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest()))
    return dict(expected=out,sources=records)
def verify(row,references):
    expected=references['expected'].get(str(row['seed']))
    if expected is None:return False
    actual={k:row.get(k)for k in FIELDS}
    if actual!=expected:raise RuntimeError('Fresh pooled original expert screen differs from retained real original screen: '+str(row['seed']))
    return True
