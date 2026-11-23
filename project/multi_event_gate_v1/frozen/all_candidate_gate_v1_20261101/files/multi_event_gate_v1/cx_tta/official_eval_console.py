"""RoboTwin eval_policy_client.py terminal formatting and valid-episode counting."""
import re


def episode_output(task, policy, task_config, checkpoint, success, successes, completed, seed, color=True):
    if type(success) is not bool or not 0 <= successes <= completed or completed < 1:
        raise ValueError('Completed Boolean outcomes and cumulative counts required')
    text = ('\033[92mSuccess!\033[0m' if success else '\033[91mFail!\033[0m') + '\n'
    text += (
        f'\033[93m{task}\033[0m | \033[94m{policy}\033[0m | '
        f'\033[92m{task_config}\033[0m | \033[91m{checkpoint}\033[0m\n'
        f'Success rate: \033[96m{successes}/{completed}\033[0m => '
        f'\033[95m{round(successes/completed*100, 1)}%\033[0m, '
        f'current seed: \033[90m{seed}\033[0m\n\n')
    return text if color else re.sub(r'\x1b\[[0-9;]*m', '', text)


def evaluate_stream(seed_start, episodes, max_attempts, screen, rollout, update, announce):
    """Count valid completed trials, including failures; never skip policy errors.

    Screen/rollout use the accepted episode index (official now_id), not the
    attempted seed offset. Persist screen records before starting the policy.
    """
    records, attempts = [], []
    for seed in range(seed_start, seed_start + max_attempts):
        row = screen(seed, len(records))
        if row['seed'] != seed or type(row['eligible']) is not bool:
            raise ValueError('Malformed eligibility record')
        attempts.append(row)
        update(attempts, records)
        if not row['eligible']:
            continue
        ep = rollout(seed, len(records), row)
        if ep['seed'] != seed or type(ep['complete']) is not bool:
            raise ValueError('Malformed rollout record')
        if ep['complete'] and type(ep['success']) is not bool:
            raise ValueError('Completed episode must have a Boolean outcome')
        records.append(ep)
        update(attempts, records)
        if not ep['complete']:
            return dict(status='smoke_censored', complete=False, attempted=len(attempts),
                        completed=len(records)-1, requested=episodes)
        announce(ep, sum(r['success'] for r in records), len(records))
        if len(records) == episodes:
            return dict(status='complete', complete=True, attempted=len(attempts),
                        completed=len(records), requested=episodes)
    raise RuntimeError(f'Attempt limit exhausted: completed {len(records)}/{episodes}; not a full evaluation')
