from pathlib import Path
import sys,subprocess,json
T=Path.cwd();sys.path.insert(0,str(T.parents[1]/'deployment'));import user_container
out=T/'outputs/native_early_next_preparation_v1';out.mkdir(parents=True,exist_ok=True);proof=out/'CPU_actual_host_exit_v1.json';assert not proof.exists()
script='/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA/multi_event_gate_v1/tta_event_v1/native_onpolicy_arm_collection_v22/scripts/cpu.sh'
cmd=user_container.command(['bash',script,'-m','native_early_next_preparation_v1.checks'],mount_project=True)
with(out/'CPU_host_v1.log').open('x')as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
proof.write_text(json.dumps(dict(exit_code=code,actual_Popen_wait_return=True,pid=p.pid,proc_absent_after_wait=not Path('/proc',str(p.pid)).exists()),indent=2)+'\n')
print('ACTUAL_CPU_EXIT',code);print('\n'.join((out/'CPU_host_v1.log').read_text().splitlines()[-15:]));raise SystemExit(code)
