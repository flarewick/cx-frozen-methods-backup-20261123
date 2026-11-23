"""Apply this runtime's frozen CPU mask before entering the existing container."""
from pathlib import Path
import argparse,json,subprocess,sys
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1]
sys.path.insert(0,str(P/'deployment'));sys.path.insert(0,str(T))
import user_container as container
from native_feedback_confirm_v5.cpu_placement import apply

def main():
 p=argparse.ArgumentParser();p.add_argument('args',nargs='*');a=p.parse_args()
 apply(json.loads((T/'outputs/native_visible_expand_initial_v2/cpu_placement_plan.json').read_text()))
 canonical=container.CANONICAL/'VLA-Adapter/cx_new_direction_guide_TTA'/C.relative_to(P)
 raise SystemExit(subprocess.run(container.command(['bash',str(canonical/'scripts/cpu.sh'),*a.args],mount_project=True)).returncode)
if __name__=='__main__':main()
