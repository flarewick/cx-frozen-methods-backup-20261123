"""Trace only received signals for a newly launched owned subtree; never attach."""
from pathlib import Path
def command(argv,output):
 return ['/usr/bin/strace','-f','--seccomp-bpf','-qq','-e','trace=none','-e','signal=SIGINT,SIGTERM,SIGHUP','-o',str(Path(output)),'--',*argv]
