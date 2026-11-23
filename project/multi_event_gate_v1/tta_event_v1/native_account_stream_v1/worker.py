"""Original collection bodies; one NVML initialization per CPU worker."""
import ast,copy,json,os,sys,time
from pathlib import Path
import pynvml as n
T=Path(__file__).resolve().parents[1]
SOURCES={'inventory':T/'native_fast_inventory_v1/worker.py','rule4':T/'native_server_rule4_v1/worker.py'}
def collector(kind):
 tree=ast.parse(SOURCES[kind].read_text());original=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='collect')
 assert len(original.body)==2 and ast.unparse(original.body[0])=='n.nvmlInit()'
 block=original.body[1];assert isinstance(block,ast.Try)and not block.handlers and not block.orelse and len(block.finalbody)==1 and ast.unparse(block.finalbody[0])=='n.nvmlShutdown()'
 changed=copy.deepcopy(original);changed.body=copy.deepcopy(block.body)
 restored=copy.deepcopy(original);restored.body[1].body=copy.deepcopy(changed.body)
 assert ast.dump(restored)==ast.dump(original),'Only NVML session lifetime may change'
 scope=dict(n=n,os=os,Path=Path,json=json,time=time)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(SOURCES[kind]),'exec'),scope)
 return scope['collect']
def main():
 collect={kind:collector(kind)for kind in SOURCES};n.nvmlInit()
 try:
  for line in sys.stdin:
   request=json.loads(line);assert set(request)=={'nonce','sequence','kind'}and request['kind']in collect
   began=time.time_ns();fleet=collect[request['kind']]()
   print(json.dumps(dict(request=request,data=fleet,worker_pid=os.getpid(),request_started_unix_ns=began,response_ready_unix_ns=time.time_ns()),allow_nan=False,separators=(',',':')),flush=True)
 finally:n.nvmlShutdown()
if __name__=='__main__':main()
