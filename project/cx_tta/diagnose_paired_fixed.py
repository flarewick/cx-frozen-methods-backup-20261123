"""Explain S2e offline gate failures without relabeling them as policy failures."""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from .probe_candidates import read
from .paths import write_json
from .audit import digest


def failures(group):
    failed=[]
    for side,a in group['arms'].items():
        for key,bound in (('angle_deg',3),('position_m',.01),('arm_rmse_rad',.05),('gripper_mae',.05),
                          ('velocity_error_rmse_rad',.03),('joint_limit_violations',0),('gripper_outside_margin',0)):
            if a[key] is not None and a[key]>bound:failed.append(side+':'+key)
        if a['pose_under_3deg_1cm_fraction']<.9:failed.append(side+':pose_fraction')
    if group['seam_error_rmse_rad'] is not None and group['seam_error_rmse_rad']>.03:failed.append('seam')
    assert (not failed)==group['recovery_fit_gate']
    return failed


def main(run):
    s=read(run/'summary.json');cells=[]
    for group in sorted({c['group'] for c in s['comparisons']}):
        for encoding in ('repeat','fourier'):
            for variant in ('raw','capped'):
                cs=[c for c in s['comparisons'] if (c['group'],c['encoding'],c['variant'])==(group,encoding,variant)]
                assert len(cs)==3
                for label in ('included','omitted'):
                    cells.append(dict(group=group,encoding=encoding,variant=variant,condition=label,
                        gates=sum(c[label]['recovery_fit_gate'] for c in cs),failures=dict(Counter(f for c in cs for f in failures(c[label]))),
                        right_mean={k:float(np.mean([c[label]['arms']['right'][k] for c in cs]))
                            for k in ('angle_deg','position_m','gripper_mae','pose_under_3deg_1cm_fraction')},
                        seam_range=[min(c[label]['seam_error_rmse_rad'] for c in cs),max(c[label]['seam_error_rmse_rad'] for c in cs)]))
    gates={variant:{label:sum(c[label]['recovery_fit_gate'] for c in s['comparisons'] if c['variant']==variant)
            for label in ('included','omitted')} for variant in ('raw','capped')}
    preservation={variant:{part:sum(r['variants'][variant][part]['preservation_gate'] for r in s['reports'])
            for part in ('anchor_train','anchor_developer')} for variant in ('raw','capped')}
    write_json(run/'decision.json',dict(status='complete_offline_diagnostic',cells=cells,gates=gates,preservation=preservation,
        counts=dict(models=24,paired_scene_encoding_seed_cells=18,independent_switch_scenes=3),
        interpretation=[
            'Own-scene supervision strongly improves right-arm pose but full fit is incomplete: raw4/18 complete gates, omit0/18.',
            'All omitted raw scene/encoding/seed cases fail seam continuity. Not solely a gate threshold artifact.',
            'Original cap fails18/18 even for included scenes. Teacher imitation distance is not proof of closed-loop infeasibility.',
            'Training anchors preserved24/24; developer cup anchor preserved0/24, for raw and capped. Offline drift is not observed policy failure.',
            'Fourier improves some fitted trajectories but does not establish unseen-scene transfer. Do not select best seed.',
            'Three scenes only. Supervision dependence does not uniquely identify memorization versus representation, coverage, or target ambiguity.'],
        next_order=[
            'Freeze current24 models and data. Audit phase/teacher-path consistency, gripper modes and within-scene fork targets using existing traces.',
            'Assess whether legal RGB/proprio/history distinguish target-relative geometry and phase across whole-scene omission. Privileged quantities diagnostic only.',
            'If justified, preregister one shared relative-geometry/arm-gripper representation change with same frozen initialization and budget; include identity preservation.',
            'Keep raw action-center and original bounded residual hypotheses separate. No automatic cap enlargement.',
            'Only after repeatable held-scene candidate benefit and preservation, test real-time compiled paired closed loop; no new collection/router/TTA now.'],
        evidence_sha256={str(run/n):digest(run/n) for n in ('protocol.json','result.json','candidate_audit.json','summary.json','metric_boundary_check.json')},
        code_sha256=digest(Path(__file__))))
    print('DECISION',gates,preservation,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('run',type=Path);args=parser.parse_args();main(args.run.resolve())
