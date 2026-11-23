"""CPU-only prepared16-core entry, using the installed environment and noCUDA."""
from pathlib import Path
import argparse,json,os,sys,subprocess
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'));import user_container as container
def main():
 p=argparse.ArgumentParser();p.add_argument('args',nargs='*');a=p.parse_args();mask=set(json.loads((T/'outputs/native_visible_expand_semantics_v2/CPU_placement.json').read_text())['logical_CPUs'])
 assert len(mask)==16 and mask<=set(os.sched_getaffinity(0));os.sched_setaffinity(0,mask)
 canonical=container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/C.relative_to(P)
 raise SystemExit(subprocess.run(container.command(['bash',str(canonical/'scripts/cpu.sh'),*a.args],mount_project=True)).returncode)
if __name__=='__main__':main()
