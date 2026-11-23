"""Audit the actual fixed twenty-scene candidate in the existing CPU runtime."""
from pathlib import Path
import subprocess,sys,json
T=Path(__file__).resolve().parents[1];P=T.parents[1]
def main():
 sys.path.insert(0,str(P/'deployment'));import user_container as container
 canonical=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA');script=canonical/(T/'native_onpolicy_coverage_v7/scripts/cpu.sh').relative_to(P)
 process=subprocess.Popen(container.command(['bash',str(script),'-m','native_object_association_v1.check_candidate'],mount_project=True));code=process.wait()
 with(T/'outputs/native_object_association_v1/CPU_candidate_actual_host_exit.json').open('x')as f:json.dump(dict(exit_code=code,actual_Popen_wait_return=True,pid=process.pid,proc_absent_after_wait=not Path('/proc',str(process.pid)).exists(),no_GPU_devices_exposed=True),f,indent=2)
 raise SystemExit(code)
if __name__=='__main__':main()
