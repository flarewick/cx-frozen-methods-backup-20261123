"""Aggregate one newly executed L0/S0/SP task triplet, retaining paired outcomes."""
import argparse
import json
from pathlib import Path

from .audit import digest
from .evaluate_b2 import TASKS
from .paths import ROOT, new_run, write_json


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare(group):
    all_runs = {}
    for condition in ('L0', 'S0', 'SP'):
        pointer = read(group / f'{condition}.json')
        run = Path(pointer['run'])
        if digest(run / 'result.json') != pointer['result_sha256'] or digest(run / 'manifest.json') != pointer['manifest_sha256']:
            raise ValueError('Changed evaluation result/manifest: ' + condition)
        manifest = read(run / 'manifest.json')
        if manifest['condition'] != condition or manifest['scope'] != 'closed_loop_development':
            raise ValueError('A full development triplet is required for success-rate aggregation')
        if not read(run / 'result.json')['complete']:
            raise ValueError('Do not aggregate an incomplete/error evaluation as a completed triplet')
        all_runs[condition] = dict(path=str(run), manifest=manifest, result=read(run / 'result.json'),
            provenance=read(run / 'provenance.json'),
            episodes={row['seed']: row for row in read(run / 'episodes.json')},
            first_queries={row['seed']: row for row in reversed(records(run / 'queries.jsonl')) if row['query'] == 0})
    baseline = all_runs['L0']
    output = dict(scope='20-seed reused development screening, not independent confirmation',
        task=baseline['manifest']['task'], runs={key: value['path'] for key, value in all_runs.items()},
        task_results={key: value['result'] for key, value in all_runs.items()}, comparisons={})
    for condition, baseline_condition, comparison_key in (('S0', 'L0', 'S0'), ('SP', 'L0', 'SP'), ('SP', 'S0', 'SP_vs_S0')):
        baseline = all_runs[baseline_condition]
        candidate = all_runs[condition]
        for key in ('task', 'training_run', 'seeds', 'projection_sha256', 'robot_config_sha256',
                    'policy_init_seed', 'rollout_protocol', 'compiled', 'task_config', 'instruction_type',
                    'noise_source', 'residual_strength', 'legacy_l0_sha256'):
            if baseline['manifest'][key] != candidate['manifest'][key]:
                raise ValueError('Triplet protocol mismatch: ' + key)
        for key in ('base_checkpoint_sha256', 'checkpoint_stat', 'normalization_sha256', 'tokenizer_sha256',
                    'code_sha256', 'runtime_sha256', 'policy_source_sha256', 'task_config_sha256'):
            if baseline['provenance'][key] != candidate['provenance'][key]:
                raise ValueError('Triplet runtime/provenance mismatch: ' + key)
        pairs = []
        for seed in baseline['manifest']['seeds']:
            b, c = baseline['episodes'][seed], candidate['episodes'][seed]
            if b.get('eligible') is not True or c.get('eligible') is not True:
                pairs.append(dict(seed=seed, comparison='invalid', baseline_eligible=b.get('eligible'),
                    candidate_eligible=c.get('eligible'), baseline_reason=b.get('reason'), candidate_reason=c.get('reason')))
                continue
            if b['instruction'] != c['instruction']:
                raise ValueError('Different instruction for paired seed ' + str(seed))
            bq, cq = baseline['first_queries'][seed], candidate['first_queries'][seed]
            initial_parity = dict(observation_equal=bq['observation_sha256'] == cq['observation_sha256'],
                rng_before_equal=bq['rng_before_query'] == cq['rng_before_query'],
                rng_after_equal=bq['rng_after_sampler'] == cq['rng_after_sampler'],
                base_equal=bq['base_chunk_sha256'] == cq['base_chunk_sha256'],
                inputs_equal=bq['structured_input_sha256'] == cq['structured_input_sha256'])
            label = 'tie' if b['success'] == c['success'] else ('rescued' if c['success'] else 'harmed')
            pairs.append(dict(seed=seed, comparison=label, baseline_success=b['success'],
                candidate_success=c['success'], first_query_parity=initial_parity))
        valid = [row for row in pairs if row['comparison'] != 'invalid']
        rescued = sum(row['comparison'] == 'rescued' for row in valid)
        harmed = sum(row['comparison'] == 'harmed' for row in valid)
        parity = all(all(row['first_query_parity'].values()) for row in valid) and bool(valid)
        output['comparisons'][comparison_key] = dict(baseline_condition=baseline_condition, candidate_condition=condition,
            paired_valid=len(valid), paired_invalid=len(pairs) - len(valid),
            baseline_success=sum(row['baseline_success'] for row in valid), candidate_success=sum(row['candidate_success'] for row in valid),
            rescued=rescued, harmed=harmed, net_success_change=rescued - harmed,
            net_success_rate_change=(rescued - harmed) / len(valid) if valid else None,
            first_query_parity_passed=parity,
            interpretation='Initial observation/RNG/base/interface parity checked; later trajectories naturally diverge. '
                'If parity fails, do not treat seed outcomes as a strict policy-only comparison.', pairs=pairs)
    output['strict_initial_pairing_passed'] = all(v['first_query_parity_passed'] for v in output['comparisons'].values())
    write_json(group / 'comparison.json', output)
    for value in output['comparisons'].values():
        n = value['paired_valid']
        print(f'{output["task"]} {value["candidate_condition"]} vs {value["baseline_condition"]}: {value["candidate_success"]}/{n} vs '
            f'{value["baseline_success"]}/{n}; rescued={value["rescued"]}, harmed={value["harmed"]}, '
            f'net={value["net_success_change"]:+d}; initial_parity={value["first_query_parity_passed"]}', flush=True)
    print('COMPARISON', group / 'comparison.json', flush=True)
    return output


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--create-group', action='store_true')
    p.add_argument('--task', choices=TASKS)
    p.add_argument('--training-run', type=Path)
    p.add_argument('--group', type=Path)
    a = p.parse_args()
    if a.create_group:
        if a.task is None or a.training_run is None:
            p.error('--create-group requires --task and --training-run')
        group = new_run('triplet', group=('evaluations', 'structured_action', a.task, 'comparisons'))
        write_json(group / 'design.json', dict(task=a.task, training_run=str(a.training_run.resolve()),
            conditions=['L0', 'S0', 'SP'], ordering='Sequential for one task/GPU; tasks may run on separate cards',
            legacy_l0='Instructions/eligibility only; L0 is rerun in the new official entry'))
        print(group)
    else:
        if a.group is None:
            p.error('--group required')
        compare(a.group.resolve())


if __name__ == '__main__':
    main()
