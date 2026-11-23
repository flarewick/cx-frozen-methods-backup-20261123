"""Independent saved-JSON branch check; no model import or fitting."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def verify_run(run):
    run=run.resolve();root=Path(__file__).resolve().parents[1]/'outputs'
    if not run.is_relative_to(root):raise ValueError('Project outputs only')
    episodes=json.loads((run/'episodes.json').read_text());traces={}
    for e in episodes:
        with gzip.open(run/'traces'/(e['trace_stem']+'.jsonl.gz'),'rt') as stream:
            traces[(e['seed'],e['condition'])]=[json.loads(s) for s in stream]
    checks=[]
    fields=('observation_sha256','structured_input_sha256','rng_before_query','rng_after_sampler','base_chunk_sha256','raw_delta_sha256','action_chunk_sha256','native_before_sha256','native_after_sha256','executed')
    for seed in sorted({e['seed'] for e in episodes}):
        for gate in ('G_candidate_cost2','G_observation_cost2'):
            rows=traces[(seed,gate)];scored=[r for r in rows if r['empirical_gate']['scored']]
            assert len(scored)<=1
            decision=scored[0]['empirical_gate'] if scored else None
            accepted=bool(decision and decision['accepted'])
            if decision:
                assert accepted==(decision['scores']['rescue']-2*decision['scores']['harm']>0)
                assert scored[0]['query']==scored[0]['gate_decision']['window'][0]
            reference='F_first_full' if accepted else 'Z_zero';target=traces[(seed,reference)]
            assert len(rows)==len(target)
            for a,b in zip(rows,target):
                assert all(a[k]==b[k] for k in fields)
                assert a['rng_after_gate']==a['rng_after_sampler']
            checks.append(dict(seed=seed,gate=gate,scored_query=None if not scored else scored[0]['query'],accepted=accepted,
                margin=None if not decision else decision['margin'],reference=reference,identical_queries=len(rows),RNG_unchanged=True))
    result=dict(passed=True,scope='Independent direct JSON trace comparison; no model or threshold refit',branches=checks,
        source_sha256={str(run/'result.json'):hashlib.sha256((run/'result.json').read_bytes()).hexdigest(),
            str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (run/'independent_branch_check.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('runs',nargs='+',type=Path);args=ap.parse_args()
    for run in args.runs:print(run,json.dumps(verify_run(run)),flush=True)

if __name__=='__main__':main()
