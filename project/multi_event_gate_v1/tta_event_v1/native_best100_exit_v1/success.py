"""Called only after the complete scientific batch and finally cleanup."""
import os,sys

def complete_and_exit():
 sys.stdout.flush()
 sys.stderr.flush()
 os._exit(0)
