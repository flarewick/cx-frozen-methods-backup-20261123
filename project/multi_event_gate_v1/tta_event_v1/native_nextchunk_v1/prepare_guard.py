"""Copy the strict guard with only source-audited normal-child retirement changes."""
import ast,hashlib,json
from pathlib import Path
C=Path(__file__).resolve().parent;T=C.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    parent=T/'resource_guard_v1/guard.py';text=parent.read_text();original=text
    patches=[]
    def replace(before,after):
        nonlocal text
        assert text.count(before)==1,before
        text=text.replace(before,after)
        patches.append(dict(before=before,after=after))
    replace('steady_window_seconds=60, prepared_process=None, before_monitor=None):',
'''steady_window_seconds=60, prepared_process=None, before_monitor=None,
                    child_exit_proof=None, child_exit_identity=None):''')
    replace("    total_mib = int(gpu_doctor.query(['-i', str(gpu), '--query-gpu=memory.total']).strip())",
'''    if (child_exit_proof is None)!=(child_exit_identity is None):
        raise ValueError('Both child-exit proof path and fresh attempt identity required')
    total_mib = int(gpu_doctor.query(['-i', str(gpu), '--query-gpu=memory.total']).strip())''')
    replace('    verified_gpu_pids = set()',
'''    verified_gpu_pids = set()
    verified_exit_identities = {}''')
    replace('                verified_gpu_pids.update(active)',
'''                verified_gpu_pids.update(active)
                from child_exit import verified_identity,retired
                pending_exit_identities = {
                    pid:verified_identity(pid,root_pid=process.pid,table=table)
                    for pid in active}''')
    priority='                priority=arrival_order.observe(elapsed,active,[] if terminal_stale else foreign)\n'
    replace(priority,'')
    replace('                for pid in active:\n                    if pid in pid_util:',
'''                child_exit_note = dict(accepted=False,reason='normal_child_exit_proof_not_configured')
                retired_children = set()
                if child_exit_proof is not None:
                    retired_children,child_exit_note=retired(
                        child_exit_proof,child_exit_identity,verified_exit_identities)
                terminal_stale=sorted(set(terminal_stale)|(set(foreign)&retired_children))
                active=[pid for pid in active if pid not in retired_children]
                priority=arrival_order.observe(
                    elapsed,active,sorted(set(foreign)-set(terminal_stale)))
                # Only a prior sample with a directly available per-PID metric
                # can authorize retirement on a later iteration. First-unseen,
                # recovered/missing or unknown PIDs never get this exemption.
                for pid in active:
                    candidate=pending_exit_identities.get(pid)
                    if candidate is not None and pid in pid_util and pid not in missing_initial:
                        previous=verified_exit_identities.get(pid)
                        if previous is not None and previous['child_started_ticks']!=candidate['child_started_ticks']:
                            raise ValueError('A verified CUDA PID was reused')
                        if previous is None:
                            verified_exit_identities[pid]=candidate
                for pid in active:
                    if pid in pid_util:''')
    replace('                             "raw_foreign_pids": sorted(foreign),',
'''                             "raw_foreign_pids": sorted(foreign),
                             "normal_child_exit_proof": child_exit_note,
                             "normal_child_retired_pids": sorted(retired_children),
                             "normal_child_previously_metric_verified_pids": sorted(verified_exit_identities),''')
    target=C/'strict_guard.py'
    assert not target.exists();target.write_text(text)
    # Reverse the complete explicit patch list and compare parsed whole module.
    reversed_text=text
    for patch in reversed(patches):
        assert reversed_text.count(patch['after'])==1 if patch['after'] else True
        if patch['after']:
            reversed_text=reversed_text.replace(patch['after'],patch['before'],1)
        else:
            # The only deleted statement was arrival.observe. Restore it at
            # the exact original pre-metric position, then undo other patches.
            needle='                pid_util=process_utilization(gpu)\n'
            assert reversed_text.count(needle)==1
            reversed_text=reversed_text.replace(needle,patch['before']+needle,1)
    assert ast.dump(ast.parse(reversed_text),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False)
    proof=dict(passed=True,parent_sha256=sha(parent),patched_sha256=sha(target),
          reverse_whitelisted_patches_whole_AST_equal=True,patches=patches,
          scope='All original resource/arrival/CPU/RSS/VRAM/utilization/deadline/release code preserved outside explicit source patches. New exemption only fresh certified normal-reaped prior-metric-verified samePID absent. No actual GPU qualification.')
    out=T/'outputs/native_nextchunk_v1/guard_source_audit.json';assert not out.exists()
    out.write_text(json.dumps(proof,indent=2)+'\n')
    print('CHILD_EXIT_GUARD_SOURCE_AUDIT_PASS',len(patches),'NO_GPU')
if __name__=='__main__':run()
