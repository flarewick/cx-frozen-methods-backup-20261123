"""Fail-closed CPU checks for one shared-sampler residual attribution run.

This module never imports a policy or a physics runtime. Exact numerical and
native-state equality is a diagnostic gate, not a success-rate improvement claim.
Differences are quantified only for queries with equal observations and RNG.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import output_path, replace_json


SHAM_CONDITIONS = ("L0", "S0_zero")
CAPS = np.asarray([.1] * 6 + [.2] + [.1] * 6 + [.2], np.float32)
STRUCTURED_FIELDS = frozenset(("visual", "text", "state", "past_visual", "past_text",
                              "past_state", "has_history", "ee_pose", "past_ee_pose"))


def _read(path):
    return json.loads(Path(path).read_text())


def _hash_array(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _exact(a, b):
    """Byte equality also detects signed-zero differences hidden by tolerance."""
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _native_hash(row, which):
    keys = {"before": ("native_before_sha256", "state_before_sha256"),
            "after": ("native_after_sha256", "state_after_sha256"),
            "complete": ("native_complete_states_sha256",)}[which]
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    for key in (("native_" + which, which) if which != "complete" else ("native_complete_states",)):
        value = row.get(key)
        if value is not None:
            if isinstance(value, dict) and value.get("sha256") is not None:
                return value["sha256"]
            return _hash_json(value)
    return None


def _first_difference(left, right, getter):
    for i, (a, b) in enumerate(zip(left, right)):
        if getter(a) != getter(b):
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def _load_trace(run, episode):
    stem = str(episode["trace_stem"])
    # Traces may be analyzed outside outputs/, but a stored stem cannot escape.
    if Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("Unsafe trace stem")
    logs = run / "traces" / (stem + ".jsonl.gz")
    with gzip.open(logs, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    with np.load(run / "traces" / (stem + ".npz"), allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ("base", "raw_delta", "applied_delta", "state")}
    return dict(episode=episode, rows=rows, arrays=arrays)


def _trace_errors(trace, *, sham):
    episode, rows, arrays = trace["episode"], trace["rows"], trace["arrays"]
    errors = []
    count = len(rows)
    if not count or episode.get("queries") != count:
        errors.append("Missing queries or log/episode query-count mismatch")
    for key, value in arrays.items():
        shape = (count, 14) if key == "state" else (count, 32, 14)
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            errors.append(f"Invalid {key} shape/dtype/finite values")
    if errors:
        return errors
    if (episode.get("initial_native_sha256") is not None
            and episode["initial_native_sha256"] != _native_hash(rows[0], "before")):
        errors.append("Episode initial native hash does not match its trace")
    if (episode.get("initial_observation_sha256") is not None
            and episode["initial_observation_sha256"] != rows[0].get("observation_sha256")):
        errors.append("Episode initial observation hash does not match its trace")
    previous_count, original_limit, total_executed = None, None, 0
    for i, row in enumerate(rows):
        def fail(message):
            errors.append(f"query {i}: {message}")
        query_index = row.get("query", row.get("chunk"))
        if type(query_index) is not int or query_index != i:
            fail("Noncontiguous query index")
        structured = row.get("structured_input_sha256")
        if (not isinstance(structured, dict) or set(structured) != STRUCTURED_FIELDS
                or any(not isinstance(value, str) or not value for value in structured.values())):
            fail("Missing exact nine-field structured observable hashes")
        zero_history = row.get("history_policy") == "zero"
        expected_history = i > 0 and not zero_history
        if (type(row.get("has_history")) is not bool or row["has_history"] != expected_history
                or "history_query" not in row
                or row.get("history_query") != (i - 1 if expected_history else None)
                or (expected_history and type(row.get("history_query")) is not int)):
            fail("Invalid causal history flag/query index")
        for key in ("observation_sha256", "rng_before_query", "rng_after_feature",
                    "rng_after_sampler", "base_chunk_sha256", "action_chunk_sha256"):
            if not isinstance(row.get(key), str) or not row[key]:
                fail("Missing " + key)
        if row.get("rng_before_query") != row.get("rng_after_feature"):
            fail("Auxiliary feature extraction consumed RNG")
        if row.get("rng_before_query") == row.get("rng_after_sampler"):
            fail("Internal sampler RNG did not advance")
        if row.get("base_infer_calls") != 1:
            fail("Expected exactly one real VLA inference")
        if row.get("obs_updates", row.get("observation_updates")) != 32:
            fail("Expected get_obs after every official chunk action")
        executed = row.get("executed")
        if type(executed) is not int or not 0 < executed <= 32:
            fail("Invalid actual native action count")
        else:
            total_executed += executed
        controls = {}
        for which in ("before", "after"):
            if _native_hash(row, which) is None:
                fail("Missing native " + which + " state diagnostic")
            native = row.get("native_" + which, row.get(which))
            if not isinstance(native, dict) or native.get("usable") is not True:
                fail("Readable native " + which + " coverage is unusable or missing")
            elif _hash_json(native) != _native_hash(row, which):
                fail("Native " + which + " diagnostic hash mismatch")
            control = native.get("control", {}) if isinstance(native, dict) else {}
            counter, limit = control.get("take_action_cnt"), control.get("step_lim")
            if (type(counter) is not int or type(limit) is not int or limit <= 0
                    or not 0 <= counter <= limit or type(control.get("eval_success")) is not bool):
                fail("Invalid native " + which + " counter/budget/terminal flag")
            else:
                controls[which] = control
        if len(controls) == 2:
            before, after = controls["before"], controls["after"]
            if original_limit is None:
                original_limit = before["step_lim"]
            if before["step_lim"] != original_limit or after["step_lim"] != original_limit:
                fail("Native budget changed within episode")
            if previous_count is None:
                if before["take_action_cnt"] != 0:
                    fail("Native rollout did not begin at action count zero")
            elif before["take_action_cnt"] != previous_count:
                fail("Native action count is not contiguous across queries")
            if (type(executed) is int
                    and after["take_action_cnt"] - before["take_action_cnt"] != executed):
                fail("Actual native counter advancement differs from executed actions")
            if before["eval_success"]:
                fail("A new query was issued after native success")
            previous_count = after["take_action_cnt"]
        if row.get("base_chunk_sha256") != _hash_array(arrays["base"][i]):
            fail("Stored base numbers do not match their diagnostic hash")
        components = row.get("observation_components", {})
        if (components.get("command_state") is not None
                and components["command_state"] != _hash_array(arrays["state"][i])):
            fail("Stored commanded state does not match its observation hash")
        for field, array_key in (("raw_max_abs_correction", "raw_delta"),
                                 ("applied_max_abs_correction", "applied_delta")):
            if field in row and row[field] != float(np.abs(arrays[array_key][i]).max()):
                fail("Stored correction numbers do not match " + field)
        if np.any(np.abs(arrays["raw_delta"][i]) > CAPS + 1e-6):
            fail("Raw correction exceeds frozen native caps")
        if np.any(np.abs(arrays["applied_delta"][i]) > CAPS + 1e-6):
            fail("Applied correction exceeds frozen native caps")
        if sham and (np.any(arrays["applied_delta"][i] != 0)
                     or row.get("action_chunk_sha256") != row.get("base_chunk_sha256")):
            fail("Zero-residual action is not exact base identity")
    final = rows[-1].get("native_after", rows[-1].get("after", {}))
    final_control = final.get("control", {}) if isinstance(final, dict) else {}
    if (type(episode.get("executed_actions")) is not int
            or episode["executed_actions"] != total_executed
            or episode["executed_actions"] != final_control.get("take_action_cnt")):
        errors.append("Episode executed_actions differs from trace sum or final native count")
    if episode.get("complete") is True:
        if type(episode.get("success")) is not bool or episode["success"] != final_control.get("eval_success"):
            errors.append("Complete episode success differs from final actual native terminal flag")
        elif not episode["success"] and final_control.get("take_action_cnt") != final_control.get("step_lim"):
            errors.append("Complete native failure did not exhaust the original action budget")
    if ("observed_native_success" in episode
            and episode["observed_native_success"] != final_control.get("eval_success")):
        errors.append("Smoke observed native success differs from final native terminal flag")
    return errors


def _comparison(left, right, label):
    a, b = left["rows"], right["rows"]
    la, ra = left["arrays"], right["arrays"]
    pair = dict(comparison=label, seed=left["episode"]["seed"],
                conditions=[left["episode"]["condition"], right["episode"]["condition"]],
                repeats=[left["episode"]["repeat"], right["episode"]["repeat"]],
                query_counts=[len(a), len(b)],
                outcomes=[left["episode"].get("success"), right["episode"].get("success")])
    first = {}
    getters = {"observation": lambda row: row.get("observation_sha256"),
               "rng_before": lambda row: row.get("rng_before_query"),
               "rng_after": lambda row: row.get("rng_after_sampler"),
               "structured_observables": lambda row: row.get("structured_input_sha256"),
               "history": lambda row: (row.get("has_history"), row.get("history_query")),
               "base": lambda row: row.get("base_chunk_sha256"),
               "action": lambda row: row.get("action_chunk_sha256"),
               "native_before": lambda row: _native_hash(row, "before"),
               "native_after": lambda row: _native_hash(row, "after"),
               "execution": lambda row: (row.get("executed"), row.get("obs_updates", row.get("observation_updates")))}
    if any(_native_hash(row, "complete") is not None for row in a + b):
        getters["native_complete_states"] = lambda row: _native_hash(row, "complete")
    for key, getter in getters.items():
        first[key] = _first_difference(a, b, getter)
    first["base_numbers"] = next((i for i in range(min(len(a), len(b)))
                                  if not _exact(la["base"][i], ra["base"][i])), None)
    if len(a) != len(b) and first["base_numbers"] is None:
        first["base_numbers"] = min(len(a), len(b))
    first["command_state"] = next((i for i in range(min(len(a), len(b)))
                                   if not _exact(la["state"][i], ra["state"][i])), None)
    if len(a) != len(b) and first["command_state"] is None:
        first["command_state"] = min(len(a), len(b))
    pair["first_divergence_query"] = first
    aligned = [i for i in range(min(len(a), len(b)))
               if a[i].get("observation_sha256") == b[i].get("observation_sha256")
               and a[i].get("rng_before_query") == b[i].get("rng_before_query")]
    if aligned:
        diff = la["base"][aligned].astype(np.float64) - ra["base"][aligned].astype(np.float64)
        metrics = dict(aligned_queries=aligned, max_abs=float(np.max(np.abs(diff))),
                       rms=float(np.sqrt(np.mean(diff ** 2))),
                       numerical_different_entries=int(np.count_nonzero(diff)),
                       byte_exact=all(_exact(la["base"][i], ra["base"][i]) for i in aligned))
    else:
        metrics = dict(aligned_queries=[], max_abs=None, rms=None,
                       numerical_different_entries=None, byte_exact=None)
    pair["base_difference_same_observation_rng"] = metrics
    pair["initial_observation_equal"] = bool(a and b and first["observation"] != 0)
    pair["initial_rng_equal"] = bool(a and b and first["rng_before"] != 0)
    pair["initial_native_equal"] = bool(a and b and first["native_before"] != 0)
    pair["outcome_equal"] = pair["outcomes"][0] == pair["outcomes"][1]
    pair["exact_full_trajectory_equal"] = bool(a and b and all(v is None for v in first.values()))
    pair["passed"] = pair["exact_full_trajectory_equal"] and pair["outcome_equal"]
    return pair


def _probe(run):
    path = run / "steady_probe.json"
    if not path.is_file():
        return dict(passed=False, reason="Missing fixed-observation restored-RNG steady probe")
    data = _read(path)
    comparisons = data.get("comparisons", data.get("pairs"))
    if data.get("passed") is not True or not isinstance(comparisons, list) or not comparisons:
        return dict(passed=False, reason="Steady probe lacks passing nonempty numerical comparisons", data=data)
    # Recompute from saved numbers. A reported boolean or max=0 cannot hide a
    # differing numerical output or the wrong restored RNG/input context.
    errors, measured = [], []
    try:
        with np.load(run / "steady_probe.npz", allow_pickle=False) as saved:
            base = saved["base"].copy()
        queries = data["queries"]
        if (base.dtype != np.float32 or base.ndim != 3 or base.shape[1:] != (32, 14)
                or base.shape[0] < 2 or not np.isfinite(base).all()
                or len(queries) != len(base) or len(comparisons) != len(base) - 1):
            raise ValueError("Invalid fixed-probe numbers/query design")
        if {q.get("condition") for q in queries} != set(SHAM_CONDITIONS):
            errors.append("Fixed-input probe must cover both sham conditions")
        if (data.get("warmup_queries") != 2 or data.get("action_attempts") != 0
                or not data.get("initial_native_sha256")
                or data["initial_native_sha256"] != data.get("final_native_sha256")):
            errors.append("Warmup/action count or fixed-probe native-state parity failed")
        for i, query in enumerate(queries):
            structured = query.get("structured_input_sha256")
            if (not isinstance(structured, dict) or set(structured) != STRUCTURED_FIELDS
                    or any(not isinstance(value, str) or not value for value in structured.values())
                    or structured != queries[0].get("structured_input_sha256")
                    or query.get("has_history") is not False or query.get("history_query") is not None):
                errors.append(f"probe {i}: fixed-input structured hashes or reset history differ")
            if (query.get("base_chunk_sha256") != _hash_array(base[i])
                    or query.get("action_chunk_sha256") != query.get("base_chunk_sha256")
                    or query.get("identity") is not True or query.get("applied_zero") is not True):
                errors.append(f"probe {i}: stored base numbers/hash or zero action identity failed")
            if (not query.get("rng_before_query") or not query.get("rng_after_sampler")
                    or query["rng_before_query"] != query.get("rng_after_feature")
                    or query["rng_before_query"] == query["rng_after_sampler"]
                    or query["rng_before_query"] != data.get("common_rng_sha256")
                    or query.get("base_infer_calls") != 1):
                errors.append(f"probe {i}: restored RNG or single inference contract failed")
            if i == 0:
                continue
            before_equal = query["rng_before_query"] == queries[0]["rng_before_query"]
            after_equal = query["rng_after_sampler"] == queries[0]["rng_after_sampler"]
            obs_equal = bool(query.get("observation_sha256") and query.get("observation_sha256") == queries[0].get("observation_sha256"))
            diff = base[i].astype(np.float64) - base[0].astype(np.float64)
            row = dict(index=i, base_exact=_exact(base[i], base[0]), base_max_abs=float(np.abs(diff).max()),
                       base_rms=float(np.sqrt(np.mean(diff ** 2))), rng_before_equal=before_equal,
                       rng_after_equal=after_equal, observation_equal=obs_equal)
            measured.append(row)
            claimed = comparisons[i - 1]
            maximum = claimed.get("base_max_abs", claimed.get("max_abs", claimed.get("max_base_abs")))
            rms = claimed.get("base_rms", claimed.get("rms"))
            exact = claimed.get("base_exact", claimed.get("byte_exact", claimed.get("equal")))
            if (exact is not row["base_exact"] or maximum != row["base_max_abs"] or rms != row["base_rms"]
                    or claimed.get("rng_before_equal") is not before_equal
                    or claimed.get("rng_after_equal") is not after_equal):
                errors.append(f"probe {i}: reported numerical/RNG comparison does not match saved evidence")
            if not (row["base_exact"] and before_equal and after_equal and obs_equal):
                errors.append(f"probe {i}: fixed-input internal-RNG sampler replay differs")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    return dict(passed=not errors, data=data, measured_comparisons=measured, errors=errors,
                reason=None if not errors else "Steady probe numerical/identity evidence failed")


def analyze_run(run):
    """Return an attribution gate; incomplete/malformed evidence never passes."""
    run = Path(run).resolve()
    report = dict(scope="shared-sampler development attribution, not an independent improvement estimate",
                  run=str(run), task=None, stage=None, gate_passed=False,
                  interface_passed=False, errors=[], comparisons=[])
    report["native_state_scope"] = "Readable query-boundary native states only; hidden solver/contact and renderer state excluded"
    try:
        manifest, episodes = _read(run / "manifest.json"), _read(run / "episodes.json")
        report["task"], report["stage"] = manifest.get("task"), manifest.get("stage")
        stage = str(manifest.get("stage"))
        if stage not in ("sham", "stage1", "1"):
            return _analyze_component_run(run, manifest, episodes, report)
        seeds, conditions, repeats = manifest["seeds"], manifest["conditions"], manifest["repeats"]
        if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
            report["errors"].append("Expected nonempty unique preregistered scene seeds")
        if tuple(conditions) != SHAM_CONDITIONS or type(repeats) is not int or repeats < 2:
            report["errors"].append("Sham requires L0/S0_zero and at least two repeats")
        if (manifest.get("shared_base_models") != 1 or manifest.get("shared_compiled_samplers") != 1
                or manifest.get("compiled") is not True or manifest.get("rollout_protocol") != "official"
                or manifest.get("noise_source") != "model_internal_torch"):
            report["errors"].append("Expected one shared model/compiled sampler and official internal-noise rollout")
        expected = {(s, r, c) for s in seeds for r in range(repeats) for c in conditions}
        actual, traces, trace_errors = set(), {}, []
        for episode in episodes:
            key = (episode["seed"], episode["repeat"], episode["condition"])
            if key in actual:
                report["errors"].append("Duplicate episode: " + repr(key))
            actual.add(key)
            try:
                trace = _load_trace(run, episode)
                errors = _trace_errors(trace, sham=True)
                trace_errors.extend(dict(seed=key[0], repeat=key[1], condition=key[2], error=e) for e in errors)
                if not errors:
                    traces[key] = trace
            except (OSError, ValueError, KeyError, TypeError) as exc:
                trace_errors.append(dict(seed=key[0], repeat=key[1], condition=key[2], error=str(exc)))
        if expected != actual:
            report["errors"].append("Episode design incomplete or contains unregistered episodes")
        report["trace_errors"] = trace_errors
        report["interface_passed"] = bool(expected and expected == actual and not trace_errors
                                          and not report["errors"])
        complete = (not manifest.get("max_chunks") and expected == actual and bool(episodes)
                    and all(e.get("complete") is True and type(e.get("success")) is bool for e in episodes))
        report["full_rollouts_complete"] = bool(complete)
        if not complete:
            report["errors"].append("Truncated, incomplete, or non-boolean full-rollout outcomes")
        for seed in seeds:
            for repeat in range(repeats):
                keys = [(seed, repeat, c) for c in SHAM_CONDITIONS]
                if all(k in traces for k in keys):
                    report["comparisons"].append(_comparison(*(traces[k] for k in keys), label="zero_residual_sham"))
            for condition in conditions:
                for repeat in range(1, repeats):
                    keys = [(seed, r, condition) for r in (0, repeat)]
                    if all(k in traces for k in keys):
                        report["comparisons"].append(_comparison(*(traces[k] for k in keys), label="within_condition_repeat"))
        report["steady_probe"] = _probe(run)
        report["gate_passed"] = bool(report["interface_passed"] and complete and report["steady_probe"]["passed"]
                                     and report["comparisons"] and all(p["passed"] for p in report["comparisons"]))
        report["interpretation"] = ("Strict zero-residual and native trajectory control passed; frozen component diagnostics may proceed."
                                    if report["gate_passed"] else
                                    "Attribution gate failed or is incomplete. Do not attribute outcome changes solely to residuals or change training yet.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    return report


def _analyze_component_run(run, manifest, episodes, report):
    """Summarize stage-2 only; historical baselines remain observational."""
    stage = str(manifest.get("stage"))
    if stage not in ("components", "stage2", "2"):
        report["errors"].append("Unknown attribution stage")
        return report
    reference = manifest.get("stage1_run")
    stage1, reference_manifest = None, None
    if reference:
        reference_manifest = _read(Path(reference) / "manifest.json")
        reference_stage = str(reference_manifest.get("stage"))
        if reference_stage in ("sham", "stage1", "1"):
            stage1 = analyze_run(Path(reference))
    report["stage1_gate_passed"] = bool(stage1 and stage1["stage"] in ("sham", "stage1", "1") and stage1["gate_passed"])
    if not report["stage1_gate_passed"]:
        report["errors"].append("Completed passing stage-1 run is required before component attribution")
    required = {"L0", "S0", "S0_quarter", "S0_no_gripper", "S0_no_history"}
    if (set(manifest["conditions"]) != required or len(manifest["conditions"]) != len(required)
            or not manifest["seeds"] or len(manifest["seeds"]) != len(set(manifest["seeds"]))
            or type(manifest["repeats"]) is not int or manifest["repeats"] < 1):
        report["errors"].append("Expected preregistered four frozen variants and same-process L0")
    if (manifest.get("shared_base_models") != 1 or manifest.get("shared_compiled_samplers") != 1
            or manifest.get("compiled") is not True or manifest.get("rollout_protocol") != "official"
            or manifest.get("noise_source") != "model_internal_torch"):
        report["errors"].append("Component comparison requires one shared compiled official internal-noise sampler")
    if reference_manifest:
        for key in ("task", "seeds", "training_run", "checkpoint_sha256", "task_config", "instruction_type"):
            if key not in manifest or key not in reference_manifest or manifest[key] != reference_manifest[key]:
                report["errors"].append("Component/stage-1 protocol mismatch: " + key)
    errors, traces = [], {}
    for episode in episodes:
        key = (episode["seed"], episode["repeat"], episode["condition"])
        if key in traces:
            errors.append("Duplicate component episode: " + repr(key))
            continue
        try:
            trace = _load_trace(run, episode)
            current = _trace_errors(trace, sham=episode["condition"] in SHAM_CONDITIONS)
            errors.extend(f"{key}: {error}" for error in current)
            if not current:
                traces[key] = trace
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{key}: {exc}")
    expected = {(s, r, c) for s in manifest["seeds"] for r in range(manifest["repeats"])
                for c in manifest["conditions"]}
    if set(traces) != expected:
        errors.append("Component episode design incomplete/invalid")
    report["trace_errors"] = errors
    report["interface_passed"] = bool(expected and set(traces) == expected and not errors and not report["errors"])
    full = (not manifest.get("max_chunks") and bool(episodes)
            and all(e.get("complete") is True and type(e.get("success")) is bool for e in episodes))
    report["full_rollouts_complete"] = bool(full)
    report["same_runtime_baseline_available"] = "L0" in manifest["conditions"]
    for condition in manifest["conditions"]:
        rows = [e for e in episodes if e["condition"] == condition]
        valid = [e for e in rows if e.get("complete") is True and type(e.get("success")) is bool]
        report.setdefault("observed_outcomes", {})[condition] = dict(successes=sum(e["success"] for e in valid),
                                                                    completed=len(valid), attempted=len(rows))
        for episode in rows:
            baseline_key = (episode["seed"], episode["repeat"], "L0")
            candidate_key = (episode["seed"], episode["repeat"], condition)
            if condition != "L0" and baseline_key in traces and candidate_key in traces:
                pair = _comparison(traces[baseline_key], traces[candidate_key], "frozen_component_observation")
                # A correction legitimately changes the trajectory. Never reuse
                # the sham equality flag as an efficacy test or causal effect.
                pair.pop("passed", None)
                report["comparisons"].append(pair)
    report["initial_component_pairing_passed"] = bool(report["comparisons"] and all(
        pair["initial_observation_equal"] and pair["initial_rng_equal"] and pair["initial_native_equal"]
        and pair["first_divergence_query"]["base_numbers"] != 0 for pair in report["comparisons"]))
    report["gate_passed"] = bool(report["stage1_gate_passed"] and report["interface_passed"] and full
                                 and report["same_runtime_baseline_available"] and report["initial_component_pairing_passed"])
    report["interpretation"] = ("Frozen component observations only; selected development seeds and one run do not establish generalization. "
                                "A historical/process-separated L0 is not a same-sampler baseline. No automatic training decision.")
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = analyze_run(a.run)
    destination = output_path(a.output or a.run / "attribution_analysis.json")
    replace_json(destination, report)
    print(json.dumps(dict(task=report["task"], stage=report["stage"], interface_passed=report["interface_passed"],
                          gate_passed=report["gate_passed"], report=str(destination)), ensure_ascii=False))


if __name__ == "__main__":
    main()
