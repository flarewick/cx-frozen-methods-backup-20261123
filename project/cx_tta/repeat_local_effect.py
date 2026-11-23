"""Bounded fixed-seed replication; no new architecture, data or hyperparameter search."""
import json
import subprocess
import sys
from pathlib import Path
from .audit import digest
from .paths import ROOT, new_run, write_json, replace_json


def main():
    output = new_run('local_effect_replication', group=('effect_learning',))
    datasets = {
        'full': ROOT / 'outputs/effect_learning/shared_pulse_dataset_20260930_122744_817988',
        'robot_only': ROOT / 'outputs/effect_learning/pulse_robot_only_dataset_20260930_123226_307211',
    }
    seed0 = {
        'full': ROOT / 'outputs/effect_learning/shared_effect_20260930_122932_253851',
        'robot_only': ROOT / 'outputs/effect_learning/shared_effect_20260930_123317_644970',
    }
    protocol = dict(status='frozen_before_runs', seeds=[0, 1, 2], epochs=80, learning_rate=.002,
                    datasets={k:dict(path=str(v), index_sha256=digest(v/'index.json')) for k,v in datasets.items()},
                    seed0_reuse={k:str(v) for k,v in seed0.items()},
                    modes=['conditioned','action_only','no_action','shuffled'],
                    split='16 original-train scene LOSO and 4 task holdouts; original validation unused',
                    gate='For each task and split report all seeds, all horizons and scene-level signs. h4 translation is a post-hoc developmental hypothesis. Retain full object module only with positive seed-wise increments over robot-only AND action-only on all three rigid primary objects; unsupported switch joint excluded from transfer claim. No aggregate winning-seed selection.',
                    stopping='Exactly four new fits of the existing 80-model suite (320 models), plus four reload audits. No hyperparameter search, collection, policy eval or TTA.',
                    code_sha256=digest(Path(__file__)), trainer_sha256=digest(ROOT/'cx_tta/train_shared_effect.py'),
                    model_sha256=digest(ROOT/'cx_tta/shared_effect_model.py'))
    write_json(output/'protocol.json', protocol)
    runs = [dict(variant=k, seed=0, run=str(v), reused=True) for k,v in seed0.items()]
    print(output, flush=True)
    for seed in (1, 2):
        for variant, dataset in datasets.items():
            if digest(dataset/'index.json') != protocol['datasets'][variant]['index_sha256']:
                raise ValueError('Dataset changed')
            log = output/f'{variant}_seed{seed}.log'
            with log.open('x') as handle:
                subprocess.run([sys.executable, '-B', '-m', 'cx_tta.train_shared_effect', '--dataset', str(dataset), '--seed', str(seed), '--epochs', '80'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            candidates = [Path(line) for line in log.read_text().splitlines() if line.startswith(str(ROOT/'outputs/effect_learning/shared_effect_'))]
            if len(candidates) != 1:
                raise ValueError('Ambiguous training output')
            run = candidates[0]
            with (output/f'{variant}_seed{seed}_audit.log').open('x') as handle:
                subprocess.run([sys.executable, '-B', '-m', 'cx_tta.audit_shared_effect', '--run', str(run)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            runs.append(dict(variant=variant, seed=seed, run=str(run), reused=False))
            replace_json(output/'progress.json', dict(status='running', runs=runs))
            print(f'COMPLETE {variant} seed={seed} {run}', flush=True)
    write_json(output/'result.json', dict(status='complete', protocol_sha256=digest(output/'protocol.json'), runs=runs))
    print('COMPLETE '+str(output), flush=True)

if __name__ == '__main__':
    main()
