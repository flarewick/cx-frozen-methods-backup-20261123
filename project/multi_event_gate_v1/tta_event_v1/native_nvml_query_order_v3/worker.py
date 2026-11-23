"""Original fresh acquisition/selection/3s transport; device reads ordered together."""
import json,sys
from native_process_sampler_v13 import worker as original
from .order import frame_function,clock_pair
original.frame=frame_function();original.clock_pair=clock_pair;original.TRACE_TO_STDERR=True
if __name__=='__main__':
 if len(sys.argv)>2 and sys.argv[2]=='--serve':
  from native_process_sampler_v13.stream import serve
  serve(int(sys.argv[1]))
 else:print(json.dumps(original.collect(int(sys.argv[1])),allow_nan=False))
