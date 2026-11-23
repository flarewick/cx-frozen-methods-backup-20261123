"""Build passive JSON encoder using existing Python/Torch headers, no GPU."""
import hashlib,json,subprocess,sysconfig
from pathlib import Path
import torch
assert not torch.cuda.is_initialized()
D=Path(__file__).resolve().parent;out=D/'.work/benefit_json';out.mkdir(exist_ok=True);binary=out/('cx_benefit_json'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-O2','-shared','-std=c++20','-fPIC','-ffp-contract=off','-fno-fast-math','-DPYBIND11_BUILD_ABI="_cxxabi1014"','-I'+sysconfig.get_path('include'),'-I'+str(Path(torch.__file__).parent/'include'),str(D/'benefit_json_writer.cpp'),'-o',str(binary)]
subprocess.run(cmd,check=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();record=dict(cpu_only=True,source_sha256=sha(D/'benefit_json_writer.cpp'),binary_sha256=sha(binary),binary=str(binary),compiler=cmd)
(out/'build.json').write_text(json.dumps(record,indent=2)+'\n');print('PASSIVE_JSON_CPU_BUILT',record['binary_sha256'])
