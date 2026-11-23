"""CPU package verification, then explicit CUDA/Vulkan/renderer smoke on target."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cpu',action='store_true');a=p.parse_args()
    import torch
    if sys.version_info[:3]!=(3,11,9):raise ValueError('Python 3.11.9 required')
    if torch.__version__!='2.7.1+cu126':raise ValueError('PyTorch/CUDA wheel differs')
    rt=Path('/data/cj/robo_pi/250010033/qfw/robotwin')
    sys.path[:0]=[str(rt/'RoboTwin/policy/pi05/src'),str(rt/'pi05_integration')]
    for name in ('numpy','scipy','transformers','augmax','openpi.policies.policy_config','register_motus_config',
                 'sapien','mplib','curobo','h5py','sentencepiece'):
        importlib.import_module(name);print('IMPORT_OK',name,flush=True)
    support=Path(importlib.import_module('transformers.models.siglip.check').__file__)
    if hashlib.sha256(support.read_bytes()).hexdigest()!='466e8eb7887ac5ccc98118a3169298d6ab65a284d834bae74751e3854dbc4245':
        raise ValueError('Original OpenPI transformer support differs')
    if not importlib.import_module('transformers.models.siglip.check').check_whether_transformers_replace_is_installed_correctly():
        raise ValueError('Original OpenPI transformer replacement check failed')
    report=dict(cpu_imports_passed=True,transformers_original_check_passed=True,python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda)
    if not a.cpu:
        if not torch.cuda.is_available():raise RuntimeError('No CUDA device exposed to container')
        # motion_gen allocates a CUDA tensor in a class default at import time;
        # importing it during docker build would require a GPU unnecessarily.
        importlib.import_module('curobo.wrap.reacher.motion_gen')
        report.update(gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                      free_total_bytes=torch.cuda.mem_get_info())
        if report['capability']!=(8,9):raise ValueError('This image was built for SM89; rebuild for a different GPU')
        x=torch.ones(16,device='cuda');assert float((x+x).sum())==32
        subprocess.run(['vulkaninfo','--summary'],check=True)
        import sapien.core as sapien
        engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
        scene=engine.create_scene();scene.add_ground(0);scene.set_ambient_light([.5,.5,.5])
        camera=scene.add_camera('deployment_smoke',64,64,1.,.01,10.)
        scene.update_render();camera.take_picture()
        rgba=camera.get_picture('Color');assert rgba.shape==(64,64,4)
        report['cuda_and_vulkan_camera_passed']=True
    print(json.dumps(report,default=list,indent=2),flush=True)


if __name__=='__main__':main()
