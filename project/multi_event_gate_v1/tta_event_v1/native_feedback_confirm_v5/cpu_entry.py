"""CPU-only container entry retaining the frozen GPU-local CPU affinity."""
from pathlib import Path
import argparse,json,subprocess,sys
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'))
import user_container as container
from cpu_placement import apply

def main():
 p=argparse.ArgumentParser();p.add_argument('--preload-proof',action='store_true');p.add_argument('args',nargs='*');a=p.parse_args()
 plan=json.loads((T/'outputs/native_feedback_confirm_v5/cpu_placement_plan.json').read_text());apply(plan)
 canonical=container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/C.relative_to(P)
 if a.preload_proof:
  assert not a.args;inner=['bash',str(canonical/'scripts/gpu.sh'),'--gpu-uuid',plan['gpu_uuid'],'--cpu-proof']
 else:inner=['bash',str(canonical/'scripts/cpu.sh'),*a.args]
 # Do not invoke user_container.main(), which chooses CPUs0-15 for generic CPU jobs.
 # No GPU driver options, devices, CUDA admission or work in this entry.
 raise SystemExit(subprocess.run(container.command(inner,mount_project=True)).returncode)
if __name__=='__main__':main()
