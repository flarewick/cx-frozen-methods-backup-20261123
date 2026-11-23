"""Read-only old-scene assertions; no policy outcomes, admission shortcuts or labels."""
import copy,gzip,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent;W=R/'multi_event_gate_v1'
FILE=W/'outputs/training_diagnostics/shared6_vs_single_20261025/evaluation_references/open_laptop.json'
OLD_ZERO=W/'outputs/task_adapter_eval/open_laptop/Z_zero/eval_20261024_112748_487903'
OLD_FULL=W/'outputs/task_adapter_eval/open_laptop/M_event_full/eval_20261024_112749_116669'
QUERY_KEYS=('query','observation_sha256','observation_components','rng_before_query','rng_after_feature','rng_after_sampler','base_chunk_sha256','token_sha256')
SCREEN_KEYS=('seed','episode_index','eligible','initial_physical_sha256','expert_plan_success','expert_success','scene_info','reason','error_type','error')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mapped(path):
    s=str(path);marker='cx_new_direction_guide_TTA/'
    return R/s.split(marker,1)[1] if marker in s else Path(s)
def prepare():
    old=json.loads((OLD_ZERO/'manifest.json').read_text());zero=json.loads((OLD_ZERO/'result.json').read_text());full=json.loads((OLD_FULL/'result.json').read_text())
    if not zero['complete'] or not full['complete'] or zero['completed']!=100 or zero['valid_seeds']!=full['valid_seeds']:raise RuntimeError('Original matched100 incomplete')
    episodes=json.loads((OLD_ZERO/'episodes.json').read_text());attempts=json.loads((OLD_ZERO/'journal.json').read_text())['attempts']
    files={str(p):sha(p) for p in [OLD_ZERO/'manifest.json',OLD_ZERO/'result.json',OLD_ZERO/'journal.json',OLD_ZERO/'episodes.json',OLD_FULL/'result.json']}
    scenes={}
    for ep in episodes:
        entries=list(ep['evidence_sha256'].items());entries=[(p,h) for p,h in entries if p.endswith('.jsonl.gz')]
        if len(entries)!=1:raise RuntimeError('Original query evidence ambiguous')
        path,expected=entries[0];path=mapped(path)
        if sha(path)!=expected:raise RuntimeError('Original query evidence changed')
        files[str(path)]=expected
        with gzip.open(path,'rt') as f:queries=[{k:q.get(k) for k in QUERY_KEYS} for q in map(json.loads,f)]
        scenes[str(ep['seed'])]=dict(seed=ep['seed'],episode_index=ep['episode_index'],instruction=ep['instruction'],initial_physical_sha256=ep['initial_physical_sha256'],queries=queries)
    payload=dict(protocol='old_open_laptop_scene_reference_no_policy_outcomes_v1',task='open_laptop',gpu_uuid=old['deployment_shared6']['gpu_uuid'],episodes=100,
        eligible_seeds=zero['valid_seeds'],screens={str(a['seed']):{k:a.get(k) for k in SCREEN_KEYS} for a in attempts},scenes=scenes,source_sha256=files,
        assertion_only=True,policy_outcomes_included=False,screen_admission_reexecuted=True,old_gpu_results_not_copied=True)
    FILE.parent.mkdir(parents=True,exist_ok=True)
    if FILE.exists() and json.loads(FILE.read_text())!=payload:raise RuntimeError('Frozen old-scene reference differs')
    if not FILE.exists():FILE.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print('CANDIDATE_OLD_SCENE_REFERENCE',json.dumps(dict(file=str(FILE),sha256=sha(FILE),scenes=len(scenes),screen_rows=len(attempts),policy_outcomes_included=False)))
    return payload

def load(verify=True):
    value=json.loads(FILE.read_text())
    if verify:
        for path,expected in value['source_sha256'].items():
            if sha(mapped(path))!=expected:raise RuntimeError('Original reference source changed')
    return value

def verify_query(reference,seed,instruction,diagnostic,intervened):
    if intervened:return False
    expected=reference['scenes'].get(str(seed))
    if expected is None or expected['instruction']!=instruction:raise RuntimeError('Original scene seed/instruction differs')
    q=diagnostic['query']
    if q>=len(expected['queries']):raise RuntimeError('Unmodified candidate exceeded original baseline query prefix')
    target=expected['queries'][q]
    changed=[k for k in QUERY_KEYS if diagnostic.get(k)!=target.get(k)]
    if changed:raise RuntimeError('Original baseline pre-intervention mismatch: '+str((seed,q,changed)))
    return True

if __name__=='__main__':prepare()
