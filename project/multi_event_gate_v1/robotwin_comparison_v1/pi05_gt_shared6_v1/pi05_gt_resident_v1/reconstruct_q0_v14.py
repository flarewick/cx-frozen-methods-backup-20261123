"""Replay the same seed before any policy action after an observed q0 mismatch."""
import hashlib
from native_policy_execution_v1.common import write

def identity_of(observation):
    return dict(
        physical_sha256=observation["physical_sha256"],
        rng_sha256=observation["rng_sha256"],
        RGB_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in observation["rgb"]],
        command_state_sha256=hashlib.sha256(observation["command_state"].tobytes()).hexdigest(),
    )

def components(state):
    dig = lambda b: hashlib.sha256(b).hexdigest()
    name, keys, pos, gauss, value = state["numpy"]
    return dict(
        python=dig(repr(state["python"]).encode()),
        numpy=dig(repr((name, pos, gauss, value)).encode() + keys.tobytes()),
        torch_cpu=dig(state["torch_cpu"].cpu().numpy().tobytes()),
        torch_cuda=[dig(x.cpu().numpy().tobytes()) for x in state["torch_cuda"]],
    )

def reconstruct(task, item, folder, branch, reference, replay, capture_rng):
    from cx_tta.recovery_contract import rng_fingerprint
    for attempt in range(3):
        observed = replay()
        if observed["native_action_count"] != 0 or task.take_action_cnt != 0:
            raise ValueError("Initial reconstruction is restricted to step zero")
        current = identity_of(observed)
        actual = observed["raw_observation_signature"]
        frozen_exact = actual == item["initial_observation_signature"]
        different = [] if reference is None else [k for k in reference if current[k] != reference[k]]
        if frozen_exact and not different:
            write(folder / (branch + "_initial_observation_match.json"), dict(
                actual=actual, expected=item["initial_observation_signature"], passed=True,
                single_native_get_obs=True, policy_float32_conversion_unchanged=True,
                attempt=attempt + 1))
            write(folder / (branch + "_initial_identity_v14.json"), dict(
                identity=current, attempt=attempt + 1, native_actions=0))
            return observed, current
        state = capture_rng()
        if rng_fingerprint(state) != observed["rng_sha256"]:
            raise ValueError("Live RNG changed before initial diagnosis")
        write(folder / f"{branch}_q0_mismatch_{attempt + 1}.json", dict(
            phase="before_any_policy_action", case_id=item["case_id"],
            actual=current, reference=reference, differing_fields=different,
            raw_observation_exact=frozen_exact, actual_raw=actual,
            rng_components=components(state), native_actions=0))
        if attempt == 2:
            raise ValueError("Exact paired q0 remained mismatched after three same-seed initializations: "
                             + repr(different))
        task.close_env(clear_cache=False)
    raise AssertionError("No initial observation")
