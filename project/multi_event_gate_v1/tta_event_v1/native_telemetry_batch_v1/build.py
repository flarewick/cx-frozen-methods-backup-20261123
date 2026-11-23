"""Compile against the existing immutable SAPIEN/pybind ABI, CPU only."""
from pathlib import Path
import hashlib,json,subprocess,sysconfig
import sapien,torch
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];O=T/'outputs/native_telemetry_batch_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not torch.cuda.is_initialized();s=Path(sapien.__file__).parent;out=O/'build';out.mkdir(parents=True,exist_ok=False)
 binary=out/('cx_tta_command_telemetry'+sysconfig.get_config_var('EXT_SUFFIX'))
 cmd=['g++','-O2','-shared','-std=c++20','-fPIC','-ffp-contract=off','-fno-fast-math','-DPYBIND11_BUILD_ABI="_cxxabi1014"','-I'+sysconfig.get_path('include'),'-I'+str(s.parent/'torch/include'),'-I'+str(s/'include'),'-I'+str(P/'deployment/.work/native_control/eigen'),'-I'+str(s/'include/physx/include'),str(C/'trace.cpp'),'-L'+str(s.parent/'sapien.libs'),'-lsapien','-Wl,-rpath,'+str(s.parent/'sapien.libs'),'-o',str(binary)]
 result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (out/'compiler.log').write_text(result.stdout)
 if result.returncode:raise RuntimeError('Existing compiler failed; inspect compiler.log')
 meta=dict(source_sha256=sha(C/'trace.cpp'),binary_sha256=sha(binary),binary=str(binary),compiler=cmd,CPU_only=True,CUDA_initialized=torch.cuda.is_initialized())
 (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print('COMMAND_TELEMETRY_CPU_EXTENSION_BUILT',binary,flush=True)
if __name__=='__main__':main()
