"""Build read-only contact exporter with the installed SAPIEN SDK, no GPU."""
import hashlib,json,subprocess,sysconfig
from pathlib import Path
import sapien
D=Path(__file__).resolve().parent;S=Path(sapien.__file__).parent;out=D/'.work/benefit_contacts';out.mkdir(exist_ok=True);binary=out/('cx_benefit_contacts'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd=['g++','-O2','-shared','-std=c++20','-fPIC','-ffp-contract=off','-fno-fast-math','-DPYBIND11_BUILD_ABI="_cxxabi1014"','-I'+sysconfig.get_path('include'),'-I'+str(S.parent/'torch/include'),'-I'+str(S/'include'),'-I'+str(D/'.work/native_control/eigen'),'-I'+str(S/'include/physx/include'),str(D/'benefit_native_contacts.cpp'),'-L'+str(S.parent/'sapien.libs'),'-lsapien','-Wl,-rpath,'+str(S.parent/'sapien.libs'),'-o',str(binary)]
subprocess.run(cmd,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();value=dict(cpu_only=True,source_sha256=sha(D/'benefit_native_contacts.cpp'),binary_sha256=sha(binary),binary=str(binary),compiler=cmd)
(out/'build.json').write_text(json.dumps(value,indent=2)+'\n');print('READ_ONLY_CONTACTS_CPU_BUILT',json.dumps(value))
