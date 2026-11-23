"""Original243 guards; CPU-finalize only after a proved already-exited GPU phase."""
import sys,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent
try:runpy.run_path(str(HERE.parent/'gate_all_candidate_headroom_v243/host243.py'),run_name='__main__')
except RuntimeError as error:
 if str(error)not in ('Additional243 stop on unresolved PID metrics after240s','Additional243 GPU headroom below40% after240s; stop before risking original33% floor','GPU occupied before our verified arrival or arrival order unverified; stopping only our job'):raise
 # Recovery itself checks actualPID exit, verified original intent and original33% metrics.
 runpy.run_path(str(HERE/'recover246.py'),run_name='__main__')
