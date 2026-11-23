"""Per-PID MAX(SM, bandwidth) lower bound, with raw component diagnostics.

An unavailable component contributes ZERO, never a device-wide substitute.
When both components are unavailable the PID remains unavailable and the
existing fail-closed guard/retries apply unchanged.
"""
import copy,subprocess,time
_last={}
def parse(text):
    values={};rows=[]
    for line in text.splitlines():
        fields=line.split()
        if len(fields)<5 or not fields[0].isdigit()or not fields[1].isdigit():continue
        gpu,pid=int(fields[0]),int(fields[1]);components=[]
        for token in fields[3:5]:
            if token.isdigit():
                value=int(token)
                if not 0<=value<=100:raise ValueError('Invalid per-process utilization')
                components.append(value)
            elif token in('-', 'N/A'):components.append(None)
            else:raise ValueError('Unexpected per-process utilization token')
        known=[value for value in components if value is not None]
        if known:values[pid]=max(values.get(pid,0),*known)
        rows.append(dict(gpu=gpu,pid=pid,sm=components[0],bandwidth=components[1],available=bool(known),partial=len(known)==1,missing_components_counted_as_zero=True))
    return values,rows
def process_utilization(index):
    global _last
    result=subprocess.run(['nvidia-smi','pmon','-i',str(index),'-s','u','-c','1'],check=True,stdout=subprocess.PIPE,text=True,timeout=3)
    values,rows=parse(result.stdout);_last=dict(unix_ns=time.time_ns(),gpu=index,components=rows)
    return values
def diagnostics():return copy.deepcopy(_last)
