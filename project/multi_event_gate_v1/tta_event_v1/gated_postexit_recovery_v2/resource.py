"""Original recorder interpreter3.14, unchanged exact raw180 numerical replay."""
import inspect,sys
from .common import O,ROOT,write
from .recover import prior

def main():
 assert sys.version_info[:2]==(3,14)
 prior()
 from native_governor180_v2.audit import replay
 source=inspect.getsource(replay)
 old='from native_policy_execution_v1.common import write,sha';new='from native_policy_execution_v1.common import sha\n write=_write';assert source.count(old)==1;source=source.replace(old,new)
 old="local=root/'attempt/governor180_raw.jsonl'";new='local=_raw_output';assert source.count(old)==1;source=source.replace(old,new)
 raw=O/'audits/attempt/governor180_raw.jsonl';raw.parent.mkdir(parents=True,exist_ok=True)
 scope=dict(replay.__globals__,_write=write,_raw_output=raw);exec(compile(source,__file__,'exec'),scope)
 value=scope['replay'](ROOT);assert value['passed'];print('ORIGINAL_HOST314_RAW180_EXACT_PASS',value['resource_samples'])
if __name__=='__main__':main()
