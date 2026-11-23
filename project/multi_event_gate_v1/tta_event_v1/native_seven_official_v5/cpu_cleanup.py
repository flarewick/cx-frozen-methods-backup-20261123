"""No-permit process cleanup is subtree-specific; never assert whole-card vacancy."""
from pathlib import Path
from .common import write

def record(process,root,known):
 assert not(root/'attempt/barrier/permit.json').exists()
 resources=root/'attempt/resources.jsonl';assert not resources.exists()or resources.stat().st_size==0
 assert process.poll() is not None and not Path('/proc',str(process.pid)).exists()
 assert all(not Path('/proc',str(pid)).exists() or Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[0]=='Z' for pid in known)
 write(root/'CPU_only_prepared_cleanup.json',dict(actual_Popen_wait_return=True,pid=process.pid,exit_code=process.returncode,verified_subtree_started_ticks=known,no_GPU_permit=True,no_card_wide_release_claim=True))
