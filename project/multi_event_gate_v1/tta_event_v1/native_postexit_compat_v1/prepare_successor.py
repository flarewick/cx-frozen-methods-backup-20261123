"""Prepare fresh producer/reader/train namespaces; no launch or freeze."""
from pathlib import Path
from native_policy_execution_v1.common import T

M = {'native_onpolicy_arm_collection_v72': 'native_onpolicy_arm_collection_v73',
     'native_prepared_FIT_collection_v27': 'native_prepared_FIT_collection_v28',
     'native_onpolicy_calibration_data_v55': 'native_onpolicy_calibration_data_v56'}
NOTE = '\nProspective post-exit compatibility version: fresh full25+25 only; original failed ARM72 remains excluded. Explicit actual elapsed9679.983 retained, default cap10800, no automatic budget extension. Post-exit helper binds original written-policy receipt and all raw controls. All fixed325/min8FIT/nonemptyVAL/old100 and independent dev/confirm gates unchanged. No GPU before complete CPU checks, explicit sufficient budget and first study freeze.\n'


def rename(text):
    for a, b in M.items():
        text = text.replace(a, b)
    return text


def host(text):
    begin = text.index('        from native_storage_capture_v1.audit import audit as audit_storage\n')
    end = text.index("        write(O / 'pilot_complete.json', report)\n", begin)
    return text[:begin] + ('        from native_postexit_compat_v1.tail import qualify\n'
                          '        report = qualify(root, spec)\n') + text[end:]


PLAN = '''  from native_FIT0_postexit_carry_v1.budget import apply as carry_postexit
  carry_postexit(v)
  v['complete_postexit_audit_chain_with_eight_lane_storage']=True
  files=[T/'outputs/native_postexit_compat_v1/CPU_v2/CPU_contract_checks_v1.json',T/'outputs/native_postexit_compat_v1/CPU_v2/actual_host_exit.json']
  for file in files:
   x=read(file);verify(x)
   if 'exit_code'in x:assert x['exit_code']==0 and x['actual_session_exit_observed']
   else:assert x['passed']and x['all_prospective_post_storage_audits_executed']
   v['input_sha256'][str(file)]=sha(file)
  for folder in[T/'native_storage_capture_v2',T/'native_postexit_compat_v1']:
   v['source_sha256'].update({str(p):sha(p)for p in folder.glob('*')if p.is_file()and p.suffix in('.py','.md')})
'''


def plan(text):
    anchor = '  from native_unread_capture_full_v1.budget import require_planned_full_batch\n'
    assert text.count(anchor) == 1
    return text.replace(anchor, PLAN + anchor)


def main():
    for old, new in M.items():
        source, destination = T / old, T / new
        assert not destination.exists(), 'Preserve existing successor'
        for p in source.rglob('*'):
            if not p.is_file() or p.suffix not in ('.py', '.sh', '.md'):
                continue
            target = destination / p.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            text = rename(p.read_text())
            if old.endswith('collection_v72') and p.name == 'host.py':
                text = host(text)
            if old.endswith('collection_v72') and p.name == 'collection_plan.py':
                text = plan(text)
            with target.open('x') as f:
                f.write(text)
            target.chmod(p.stat().st_mode & 0o777)
    scope = T / 'native_prepared_FIT_collection_v28/scope.py'
    scope.write_text('''from native_prepared_FIT_collection_v27.scope import transform as previous_transform, restore_runtime
from native_postexit_compat_v1.prepare_successor import rename, host, plan
def transform(name,text):
 text=rename(previous_transform(name,text))
 if name=='host.py':text=host(text)
 if name=='collection_plan.py':text=plan(text)
 return text
''')
    reader = T / 'native_onpolicy_calibration_data_v56/reader.py'
    text = reader.read_text()
    old = "   assert spec['stage_total_GPU_budget_seconds']==10800 and 7200<=spec['inherited_stage_GPU_seconds']<10800\n"
    assert text.count(old) == 1
    text = text.replace(old, """   assert spec['inherited_stage_GPU_seconds']==9679.983 and spec['stage_total_GPU_budget_seconds']in(10800,14400)
   assert spec['prior_CPU_postexit_failed_batch_excluded']and spec['actual_ARM72_cleanup_seconds_preserved']
   if spec['stage_total_GPU_budget_seconds']==14400:
    from native_FIT0_postexit_carry_v1.budget import AUTH
    authorization=read(AUTH);verify(authorization)
    assert authorization['explicit_user_authorization']and authorization['approved_total_seconds']==14400 and authorization['all_prior_GPU_seconds_retained']==9679.983
""")
    anchor = "  if spec.get('version')=='native_onpolicy_arm_collection_v73':\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "   assert spec['complete_postexit_audit_chain_with_eight_lane_storage']and qual['complete_eight_lane_postexit_chain']\n")
    reader.write_text(text)
    build = T / 'native_onpolicy_calibration_data_v56/build_host.py'
    text = build.read_text().replace('from .dataset import BASE', 'from .dataset import BASE,ARM_BASE')
    old = "assert(T/'outputs/native_onpolicy_arm_collection_v4'/(split+'_complete.json')).exists()"
    assert text.count(old) == 1
    text = text.replace(old, "assert any((T/'outputs'/v/(split+'_complete.json')).exists()for v in ARM_BASE)")
    build.write_text(text)
    source, destination = T / 'native_onpolicy_calibration_fit_v2', T / 'native_onpolicy_calibration_fit_v3'
    assert not destination.exists()
    for p in source.rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.sh', '.md'):
            target = destination / p.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            text = p.read_text().replace('native_onpolicy_calibration_fit_v2', 'native_onpolicy_calibration_fit_v3').replace('native_onpolicy_calibration_data_v28', 'native_onpolicy_calibration_data_v56')
            with target.open('x') as f:
                f.write(text)
            target.chmod(p.stat().st_mode & 0o777)
    for name in M.values():
        with (T / name / 'AGENTS.md').open('a') as f:
            f.write(NOTE)
    with (destination / 'AGENTS.md').open('a') as f:
        f.write('\nV3 only remaps the previously unrun fixed training recipe to strict Reader56. No original fit budget reset, model or optimizer before actual-build0 qualified325 plus old100. CPU absent-data check first.\n')
    print('SUCCESSOR73_READER56_FIT3_PREPARED_NOT_FROZEN_NOT_LAUNCHED', flush=True)


if __name__ == '__main__':
    main()
