"""Use real failed-batch complete controls without modifying their frozen files."""
import copy,json,types
from pathlib import Path
from native_seven_task_gate_expansion_v1.contract import T,read,mapped,sha

def main():
    from gated_handoff_compare_v1.control import compare,original
    from gated_native_audit_v1.replay import gate_record
    from declared_reference_audit_v1.expected import validate,expected
    assert compare.__code__ is original.__code__
    b=T/'outputs/native_seven_gated_dev_v1/dev_gated';actual=read(b/'actual_host_wait_exit.json');assert actual['exit_code']==1
    root=mapped(read(b/'execution_state.json')['root']);spec=read(root/'protocol.json')
    assert 'native_q0_task_handoff_v3/control.py'in(root/'GPU_host.log').read_text()
    release=json.loads((root/'attempt/resources.jsonl').read_text().splitlines()[-1]);assert release['own_gpu_released']and release['cleanup_seconds_after_start']==918.294
    captured=[]
    def capture(path,value):captured.append(dict(original_intended_path=str(path),proof=value))
    check=types.FunctionType(compare.__code__,dict(compare.__globals__,write=capture),compare.__name__,compare.__defaults__,compare.__closure__)
    for result in sorted(root.glob('cases/*/handoff_reference/result.json')):
        folder=result.parent.parent
        if(folder/'M_event_full/result.json').exists():check(folder,spec)
    assert captured and all(x['proof']['passed']for x in captured)
    first=read(next(root.glob('cases/*/M_event_full/result.json')))['queries'][0]['policy'];bad=copy.deepcopy(first);bad['gate_policy']['gate_readout_sha256']='0'*64
    try:gate_record(bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupt gate sidecar accepted')
    p=T/'outputs/native_seven_plain_dev_v2/dev_plain';r=mapped(read(p/'execution_state.json')['root']);accepted=[read(f)for f in(r/'accepted').glob('*.json')]
    wanted=expected([],accepted,set(),mandatory_first_case=True);assert len(wanted)==3
    assert validate(list(wanted),[],accepted,set(),mandatory_first_case=True)==wanted
    for wrong in (list(wanted)[:-1],list(wanted)+['unrequested:0'],list(wanted)+[next(iter(wanted))]):
        try:validate(wrong,[],accepted,set(),mandatory_first_case=True)
        except AssertionError:pass
        else:raise AssertionError('Missing/duplicate/unrequested native reference accepted')
    assert expected([],accepted,set(),mandatory_first_case=False)==set()
    regen=next(x['case_id']for x in accepted if x['episode_index']==1)
    assert expected([],accepted,{regen},mandatory_first_case=True)==wanted|{regen}
    out=T/'outputs/resource_repair_audits_v1';out.mkdir(exist_ok=False)
    value=dict(passed=True,real_complete_failed_G_controls_rechecked=captured,original_compare_CodeType_unchanged_only_audit_binding_fixed=True,frozen_old_results_modified=False,old_failed_batch_NOT_promoted=True,gate_sidecar_corruption_rejected=True,exact_declared_mandatory_and_regeneration_reference_sets_verified=True,missing_duplicate_and_unrequested_references_rejected=True,actual_GPU_started=False,source_sha256={str(f):sha(f)for d in (Path(__file__).parent,T/'gated_handoff_compare_v1',T/'declared_reference_audit_v1')for f in d.glob('*')if f.is_file()})
    (out/'CPU_audit_repair_v1.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(passed=True,actual_complete_reference_pairs=len(captured),no_old_result_changes=True,not_GPU_or_whole_batch_qualification=True)))

if __name__=='__main__':main()
