import importlib,subprocess,sys,sysconfig,threading,time
from pathlib import Path
import numpy as np,sapien,sapien.render as sr,torch
from native_policy_execution_v1.common import T,write,sha

def main():
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 source=Path(__file__).parent/'readback_owned_v3.cpp';sdk=Path(sapien.__file__).parent;O=T/'outputs/native_image_readback_gil_v1';build=O/'build_v3';build.mkdir(parents=True,exist_ok=False)
 binary=build/('cx_tta_original_image_readback'+sysconfig.get_config_var('EXT_SUFFIX'))
 cmd=['g++','-O2','-shared','-std=c++20','-fPIC','-ffp-contract=off','-fno-fast-math','-DPYBIND11_BUILD_ABI="_cxxabi1014"','-D_GLIBCXX_USE_CXX11_ABI=0','-I'+sysconfig.get_path('include'),'-I'+str(sdk.parent/'torch/include'),'-I'+str(sdk/'include'),'-I'+str(T.parents[1]/'deployment/.work/native_control/eigen'),'-I'+str(sdk/'include/physx/include'),str(source),'-L'+str(sdk.parent/'sapien.libs'),'-lsapien','-Wl,-rpath,'+str(sdk.parent/'sapien.libs'),'-o',str(binary)]
 result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(build/'compiler.log').write_text(result.stdout);assert result.returncode==0,result.stdout[-3000:]
 sys.path.insert(0,str(build));module=importlib.import_module('cx_tta_original_image_readback');assert module.registered_camera_type()is sr.RenderCameraComponent
 assert module.original_color_type()in('f4','<f4','=f4')
 for value in(np.arange(8*10*4,dtype=np.float32).reshape(8,10,4),np.arange(8*10*4,dtype=np.float32).reshape(8,10,4)[::2,1::2]):
  owned=module.CPU_owned_fixture(value);expected=value.copy();assert owned.tobytes()==expected.tobytes()and owned.strides==value.strides and not np.shares_memory(owned,value)
  value.fill(-9);assert owned.tobytes()==expected.tobytes()
 base=np.arange(8*10*4,dtype=np.float32).reshape(8,10,4)
 for value in(base,base[::2,1::2],base[::-1]):
  view=module.CPU_view_fixture(value)
  assert view.shape==value.shape and view.strides==value.strides and view.dtype==value.dtype and view.tobytes()==value.tobytes()and np.shares_memory(view,value)
 try:module.read([])
 except ValueError:pass
 else:raise AssertionError('Empty camera input accepted')
 entered=threading.Event();done=threading.Event()
 def waiting():entered.set();module.CPU_wait_fixture(100);done.set()
 thread=threading.Thread(target=waiting);thread.start();assert entered.wait(2)
 progress=0;start=time.monotonic()
 while not done.is_set():progress+=1;time.sleep(.001)
 thread.join(2);assert not thread.is_alive()and progress>=10
 assert not torch.cuda.is_initialized()
 header=sdk/'include/sapien/sapien_renderer/camera_component.h'
 write(O/'CPU_build_checks_v3.json',dict(passed=True,CPU_only=True,installed_original_camera_ABI_identity=True,original_SDK_RGBA_type=module.original_color_type(),actual_production_owned_copy_not_overwritten_by_later_input=True,actual_CPU_view_shape_strides_bytes_lifetime_preserved=True,actual_GIL_release_concurrent_CPU_progress=progress,no_GPU_camera_or_renderer_created=True,no_native_pixel_or_sustained_qualification=True,binary=str(binary),binary_sha256=sha(binary),compiler=cmd,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()},input_sha256={str(header):sha(header)}))
 print('ORIGINAL_IMAGE_GETTER_CPU_ABI_VIEW_GIL_RELEASE_PASS',progress,flush=True)
if __name__=='__main__':main()
