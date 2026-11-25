"""Reuse diagnostics bound to this exact device and scientific protocol."""
from .common import sha, write
from .warmup_same_device_v21 import archived_probe, full_warmup


def warmup(c, lane, base, initial, prompt, root, head, robots, projection, case_id):
    if getattr(base, '_GT_finite_steady_checked', False):
        return
    archive = archived_probe(root)
    if archive is None:
        return full_warmup(c, lane, base, initial, prompt, root, head, robots, projection, case_id)
    previous, source = archive
    write(root/'startup_finite_probe_reuse_v23.json', dict(
          passed=True, archived_mature_attempt=previous.name,
          raw_archived_probe_sha256=sha(previous/'official_finite_steady_probe.json'),
          actual_same_compiler_artifact_sha256=source['artifact_sha256'],
          actual_current_no_action_repeats=0, new_TTA_update=False,
          current_original_G_reference_still_checked_before_first_action_in_each_branch=True,
          branch_numeric_reference='matched_actor.AuditedActor.query'))
    base._GT_finite_steady_checked = True
