"""Stop only our own job when the fresh same-card inventory gains another PID."""
def reason(row):
 owners={int(p)for p in row.get('gpu_pids',[])}
 if not owners:return None
 foreign={str(p):v for p,v in row.get('process_memory_mib',{}).items()if int(p)not in owners}
 if foreign:return 'External same-card GPU process appeared; stop own subtree before further allocation: '+str(foreign)
 return None
