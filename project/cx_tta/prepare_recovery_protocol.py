"""Prepare a source-pinned terminal-continuation protocol; NEVER collect data."""
import hashlib
import json
from pathlib import Path
from .audit import digest
from .paths import ROOT, new_run, write_json


def rank(key):
    return hashlib.sha256(key.encode()).hexdigest()


def main():
    split_path=ROOT/'configs/intervention_batch1_20260927.json'
    split=json.loads(split_path.read_text());scenes={(s['task'],s['seed']):s for s in split['scenes'] if s['split']=='train'}
    points={}
    for family in ('cartesian_pulse_batches','short_window_batches','intervention_batches'):
        for path in sorted((ROOT/'outputs'/family).glob('*/*/index.json')):
            index=json.loads(path.read_text())
            if index['status']!='complete':raise ValueError('Incomplete prior batch')
            for job in index['records']:
                scene=(job['task'],job['seed']);key=(*scene,job['chunk'])
                if scene not in scenes or job['split']!='train' or key in points:continue
                run=Path(job['run']);manifest=json.loads((run/'manifest.json').read_text())
                result=json.loads((run/'result.json').read_text())
                if result['status']!='complete' or not result['source_observation_exact_match']:continue
                source=Path(manifest['source']);trace=source/'traces'/f'r0_seed{job["seed"]}.jsonl'
                if digest(trace)!=manifest['source_trace_sha256'] or digest(source/'manifest.json')!=manifest['source_manifest_sha256']:
                    raise ValueError('Changed source')
                sm=json.loads((source/'manifest.json').read_text());ep=next(e for e in json.loads((source/'episodes.json').read_text()) if e['seed']==job['seed'])
                steps=[json.loads(line) for line in trace.read_text().splitlines()]
                if not ep['eligible'] or ep['success']!=scenes[scene]['source_l0_success'] or sm['args']['condition']!='L0' or sm['args']['rollout_protocol']!='official':raise ValueError('Ineligible scene')
                step=steps[job['chunk']]
                if step['executed']!=32 or step['noise_source']!='model_internal_torch':continue
                points[key]=dict(task=job['task'],seed=job['seed'],chunk=job['chunk'],source=str(source),
                    source_success_stratum=ep['success'],source_trace_sha256=digest(trace),
                    source_manifest_sha256=digest(source/'manifest.json'),source_config_sha256=digest(source/'resolved_config.json'),
                    previous_replay_run=str(run),previous_replay_result_sha256=digest(run/'result.json'),
                    checkpoint=str(run/'checkpoint.npz'),checkpoint_sha256=digest(run/'checkpoint.npz'),
                    observation_signature=step['observation_components'],instruction=ep['instruction'],
                    prefix_action_count=sum(s['executed'] for s in steps[:job['chunk']]))
    chosen=[]
    for task in sorted({t for t,s in scenes}):
        for success in (False,True):
            eligible=sorted({(t,s) for t,s,c in points if t==task and scenes[t,s]['source_l0_success']==success},key=lambda pair:rank(f'{pair[0]}:{pair[1]}'))
            if not eligible:raise ValueError('No original-train scene in required stratum')
            scene=eligible[0];available=sorted([p for (t,s,c),p in points.items() if (t,s)==scene],key=lambda p:p['chunk'])
            chosen.append(available[len(available)//2])
    baseline=ROOT/'outputs/effect_learning/reachable_loss_pilot_20260930_152718_426535'
    model=baseline/'seed0_reachable_mse.pt'
    semantics=Path(json.loads((baseline/'protocol.json').read_text())['source_run'])
    run=new_run('recovery_protocol',group=('reports','protocols'))
    protocol=dict(status='design_only_not_executable',selection='Original-train only; one historically failed and one successful scene per task selected by stable scene hash, then middle existing full-chunk checkpoint. Do not inspect candidate outcomes to choose a point. Source success is a sampling stratum, not a new candidate label.',
        split_source=str(split_path),split_sha256=digest(split_path),points=chosen,
        candidate_model=dict(path=str(model),sha256=digest(model),seed_selection='Fixed seed 0 convention, not best held score',preprocessing=str(semantics/'semantic_preprocess.npz'),preprocessing_sha256=digest(semantics/'semantic_preprocess.npz')),
        candidates=['base','base_plus_shared_residual','base_minus_shared_residual'],
        action_contract='At matched checkpoint, query compiled official VLA once with internal Torch noise. Compute shared residual from current/previous observable joint-prefix features and that same base chunk. Apply candidate only to this 32-action chunk, within unchanged per-channel caps, then use unmodified L0 until native termination or original full-episode action limit. Observe after EVERY action. Do not keep replaying source actions after the fork. Teacher/GT/future labels never feed the proposal.',
        rng_contract='For each repeat, fresh predetermined policy seed at fork; compute common base and proposal once, save post-query Python/NumPy/Torch CPU+CUDA RNG states. Restore identical states and reset policy caches for each branch after exact physical-prefix replay. Internal sampling remains internal; no explicit noise tensor. This defines a new paired continuation distribution, not bitwise reproduction of the historical future.',
        continuation_policy_seeds=[910000,910001,910002],
        budget=dict(main_points=len(chosen),main_terminal_continuations=len(chosen)*3*3,
            main_additional_base_sham_continuations=len(chosen),local_gate='Before main pilot: one hash-selected failed training checkpoint, base/sham/plus/minus for two policy seeds (8 continuations). Gate data excluded from later main-pilot preference estimates if rerun.',
            action_limit='Read original task runtime/config full-episode limit and subtract executed prefix; equal remaining action budget for all branches. Never reset the budget at the fork.'),
        labels=dict(endpoint='Only native terminal success/failure or full-budget exhaustion produces an outcome. Partial horizon, crash or interrupted run is censored/missing, not a negative.',
            comparison='For identical state and policy seed, compare candidate to NEW paired base continuation. Report rescued/harmed/tied/incomplete; historical source failure/success is not the comparator. Preserve every tie and negative result.',
            preference='Store all repeat-wise outcomes and empirical win/tie/loss counts; three repeats do not justify confident hard pseudo-labels. Keep uncertain points masked for preference training until a predeclared repeatability criterion with adequate evidence is met.',
            objective='Test whether state-matched useful direction preferences exist; do not substitute object displacement, action MSE or future expert matching for task progress.',
            holdout='Whole scene and all policy seeds/branches remain in one split; original scene and demonstration validation remain sealed. This 8-point pilot is feasibility, not training scale or a generalization result.'),
        acceptance_gates=['Exact RGB/raw command and tolerance-checked physical state at branch start',
            'Base versus sham match observations/actions/native terminal outcome under matched RNG; otherwise invalidate candidate comparisons',
            'Compiled policy and observable-feature extraction do not change base action/RNG; measure overhead',
            'One-bounded-chunk intervention then unmodified L0; identical remaining native budget',
            'Stream observation/action trace and native outcome with hashes; no MP4, keep training-required frames/state',
            'Collect all branches and all signs; no only-success selection; distinguish collector error from real environment terminal failure'],
        not_implemented=['Full-terminal continuation runner','Matched runtime joint-prefix/history inference integration','Post-fork policy RNG and sham-equivalence validation','Native budget/termination wiring and censored-outcome audit'],
        authorization='Protocol preparation only. No collector/evaluation/TTA is started by this module. Current instruction remains reuse existing data; any future local verification is a separately explicit step.',
        code_sha256=digest(Path(__file__)))
    write_json(run/'protocol.json',protocol)
    print(json.dumps(dict(output=str(run),points=[{k:p[k] for k in ('task','seed','chunk','source_success_stratum')} for p in chosen],budget=protocol['budget']),indent=2),flush=True)


if __name__=='__main__':main()
