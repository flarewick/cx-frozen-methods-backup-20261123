"""One CPU fit budget, actual child wait and isolated typed finalization."""
from pathlib import Path
import os,sys,subprocess,uuid,time
from native_policy_execution_v1.common import T,read,write
P=T.parents[1];O=T/'outputs/native_onpolicy_calibration_fit_v7'
def main():
 manifest=T/'outputs/native_onpolicy_calibration_data_v60/qualified_dataset.json';assert manifest.exists()and read(manifest)['actual_build_host_exit0'];assert not list(O.glob('host_run_*'))and not(O/'runs').exists(),'Never reset fitting budget'
 assert not(T/'outputs/native_onpolicy_calibration_fit_v1/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v1').glob('host_run_*')),'Prior v1 fitting budget must not be reset'
 assert not(T/'outputs/native_onpolicy_calibration_fit_v2/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v2').glob('host_run_*')),'Prior v2 fitting budget must not be reset'
 assert not(T/'outputs/native_onpolicy_calibration_fit_v3/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v3').glob('host_run_*')),'Prior v3 fitting budget must not be reset'
 assert not(T/'outputs/native_onpolicy_calibration_fit_v4/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v4').glob('host_run_*')),'Prior v4 fitting budget must not be reset'
 assert not(T/'outputs/native_onpolicy_calibration_fit_v5/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v5').glob('host_run_*')),'Prior v5 fitting budget must not be reset'
 assert not(T/'outputs/native_onpolicy_calibration_fit_v6/runs').exists()and not list((T/'outputs/native_onpolicy_calibration_fit_v6').glob('host_run_*')),'Prior v6 fitting budget must not be reset'
 sys.path.insert(0,str(P/'deployment'));import user_container as container
 canonical=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA');script=canonical/(T/'native_onpolicy_collection_v21/scripts/cpu.sh').relative_to(P);folder=O/('host_run_'+uuid.uuid4().hex);folder.mkdir(parents=True)
 command=container.command(['bash',str(script),'-m','native_onpolicy_calibration_fit_v7.train','--run'],mount_project=True);start=time.monotonic()
 with(folder/'train.log').open('x')as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);write(folder/'launch.json',dict(pid=process.pid,command=command,no_GPU_devices_exposed=True));code=process.wait()
 write(folder/'exit.json',dict(exit_code=code,actual_Popen_wait_return=True,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists(),pid=process.pid,seconds=time.monotonic()-start));assert code==0,'Failed fit budget is retained, no reset'
 final=container.command(['bash',str(script),'-m','native_onpolicy_calibration_fit_v7.finalize','--host-run',str(canonical/folder.relative_to(P))],mount_project=True);code=subprocess.run(final).returncode;write(folder/'finalize_exit.json',dict(exit_code=code,actual_subprocess_wait_return=True));raise SystemExit(code)
if __name__=='__main__':main()
