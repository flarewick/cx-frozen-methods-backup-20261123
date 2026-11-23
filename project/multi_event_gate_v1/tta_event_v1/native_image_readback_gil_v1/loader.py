import importlib.util,sys
from native_policy_execution_v1.common import T,read,sha
from native_onpolicy_recovery_v3.common import mapped
_module=None
def load():
 global _module
 if _module is not None:return _module
 from native_direct_frames_v2.verify import verify_proof
 proof=T/'outputs/native_image_readback_gil_v1/CPU_build_checks_v3.json';value=read(proof);assert value['passed']and verify_proof(value)
 actual=read(proof.with_name('CPU_actual_host_exit_v3.json'));assert actual['exit_code']==0 and actual['actual_Popen_wait_return']and actual['proc_absent_after_wait']
 path=mapped(value['binary']);assert sha(path)==value['binary_sha256']
 name='cx_tta_original_image_readback';spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[name]=module
 import sapien.render as sr
 assert module.registered_camera_type()is sr.RenderCameraComponent
 _module=module;return module

def read(cameras):return load().read(cameras)
