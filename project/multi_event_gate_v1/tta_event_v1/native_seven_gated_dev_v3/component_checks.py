from pathlib import Path
"""Existing native pixel/prefix helpers: byte conversions, dispatch, oracle failures."""
import ast,threading,types,weakref
import numpy as np,torch
from .common import T,P,read,sha

def pixels():
 from . import readback
 from .pixel_shadow import configure,compare,report
 class CPUOverlap:
  def __init__(self):self.closed=False;configure(self)
  def run(self,fn,args):return fn(*args)
 # Compile exactly the original installed camera method, without constructing cameras.
 source=P.parent/'artifacts/rootfs/qfw/robotwin/RoboTwin/envs/camera/camera.py'
 if not source.exists():source=Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/envs/camera/camera.py')
 node=next(n for n in ast.walk(ast.parse(source.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='get_rgba');ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
 class Buffer:
  def __init__(self,n):self.value=(np.arange(48,dtype=np.float32).reshape(3,4,4)+n)/255.;self.reads=0
  def get_picture(self,name):assert name=='Color';self.reads+=1;return self.value.copy()
 class Cameras:
  collect_wrist_camera=True;collect_head_camera=True;get_rgba=ns['get_rgba']
  def __init__(self):self.left_camera=Buffer(0);self.right_camera=Buffer(10);self.static_camera_list=[Buffer(20),Buffer(30)];self.static_camera_name=['head_camera','front_camera']
 class TaskA:render_freq=0;eval_video_path=None;data_type={}
 class TaskB(TaskA):pass
 a=TaskA();a.cameras=Cameras();c=a.cameras;overlap=CPUOverlap();overlap.image_shadow['observations']=12
 old=readback.batch_read;calls=[];readback.batch_read=lambda cameras:(calls.append(len(cameras)),[cam.value.copy()for cam in cameras])[1]
 try:
  readback.install(a,True,overlap);getter=c.get_rgba;readback.install(a,True,overlap);assert c.get_rgba is getter
  for _ in range(3):
   expected=Cameras.get_rgba(c);actual=c.get_rgba();assert actual.keys()==expected.keys()
   assert all(actual[k]['rgba'].tobytes()==expected[k]['rgba'].tobytes()for k in expected)
   c.left_camera.value+=np.float32(1/255.)
  c.static_camera_list=c.static_camera_list[:1];c.static_camera_name=['head_camera'];c.get_rgba();assert calls[-1]==3
  readback.install(a,False,overlap);assert 'get_rgba'not in c.__dict__
  a.data_type={'depth':True}
  try:readback.install(a,True,overlap)
  except AssertionError:pass
  else:raise AssertionError('Unsupported modality accepted')
 finally:readback.batch_read=old
 overlap=CPUOverlap();a.data_type={};b=TaskB();b.cameras=Cameras();cams=[c.left_camera,c.right_camera,*c.static_camera_list];values=[cam.value.copy()for cam in cams]
 for _ in range(9):compare(overlap,a,cams,values)
 assert overlap.image_shadow['observations']==6
 for _ in range(9):compare(overlap,b,cams,values)
 assert overlap.image_shadow['observations']==12;overlap.closed=True;assert report(overlap)['raw_camera_arrays']==36
 broken=CPUOverlap();bad=[v.copy()for v in values];bad[0].flat[0]+=np.float32(1.)
 try:compare(broken,a,cams,bad)
 except AssertionError:pass
 else:raise AssertionError('Different raw pixels accepted')
 import sapien, sapien.render as sr
 from native_image_readback_gil_v1.loader_validated import load
 assert load().registered_camera_type()is sr.RenderCameraComponent and not torch.cuda.is_initialized()
 return True

def prefix():
 from .prefix import Prefix
 from native_policy_execution_v1 import actor
 import benefit_prefix_graph
 saved=actor.batched_encode;cache=benefit_prefix_graph._CACHE;benefit_prefix_graph._CACHE={}
 class Model:pass
 class Base:
  def __init__(self):self._model=Model()
 base=Base();data=np.zeros((3,28672),np.float16);meta={'tokens':[1,2,3]};counts={'eager':0,'fast':0,'await':0}
 def eager(*args):counts['eager']+=1;return data.copy(),dict(meta)
 actor.batched_encode=eager
 class RNG:
  lane=types.SimpleNamespace(name='open_laptop:0_M_event_full_encoder')
  def current(self):return self.lane
 c=types.SimpleNamespace(rng=RNG())
 class CUDA:
  @staticmethod
  def get_rng_state():return torch.ones(2,dtype=torch.uint8)
 class Torch:
  cuda=CUDA();get_rng_state=staticmethod(torch.get_rng_state);equal=staticmethod(torch.equal)
 class Overlap:
  def await_renderer(self):counts['await']+=1
 denoise=types.SimpleNamespace(entry=None);bridge=None
 try:
  bridge=Prefix(c,Torch(),base,denoise,Overlap(),'CPU-fixture-not-GPU')
  def fast(*args):
   counts['fast']+=1
   benefit_prefix_graph._CACHE['fixture']=dict(model=weakref.ref(base._model),owner=threading.get_ident(),captured_embedding_method_owner=object())
   return data.copy(),dict(meta)
  bridge.fast=fast
  bridge.encode(base,None,None);assert counts['fast']==0
  denoise.entry=object()
  for role in('M_event_full','TTA','baseline','zero_update','shadow_update'):
   c.rng.lane.name='open_laptop:0_'+role+'_encoder';bridge.encode(base,None,None)
  c.rng.lane.name='open_laptop:0_handoff_reference_encoder';bridge.encode(base,None,None)
  assert counts['fast']==5 and counts['await']==1 and bridge.references==1 and bridge.tasks['open_laptop']==3
  assert bridge.report()['original_feature_batch3_and_action_batch1_preserved']
  c.rng.lane.name='grab_roller:0_TTA_encoder'
  def different(*args):v=data.copy();v.flat[0]=1;return v,dict(meta)
  bridge.fast=different
  try:bridge.encode(base,None,None)
  except AssertionError:pass
  else:raise AssertionError('Different graph features accepted')
 finally:actor.batched_encode=saved;benefit_prefix_graph._CACHE=cache
 assert not torch.cuda.is_initialized();return True

def source_exact():
 from .common import C
 a=(C/'prefix.py').read_text().replace("primary=name.endswith(('_M_event_full_encoder','_TTA_encoder','_baseline_encoder','_zero_update_encoder','_shadow_update_encoder'))","primary=name.endswith('_M_event_full_encoder')")
 assert a==(T/'native_guide_reuse_v1/prefix.py').read_text()
 a=(C/'readback.py').read_text().replace('from native_image_readback_gil_v1.loader_validated import read as batch_read','from .loader_validated import read as batch_read').replace('from native_seven_gated_dev_v3.pixel_shadow import compare','from .shadow import compare')
 assert a==(T/'native_image_readback_gil_v1/readback_shadow_v2.py').read_text()
 return True


def typechecks():
 import sys,copy
 from adapter_cache_cpu import RT
 sys.path[:0]=[str(RT/'RoboTwin/policy/pi05/src'),str(RT/'pi05_integration')]
 from native_direct_frames_v1.frames import install
 install()
 from openpi.models.model import Observation
 import jaxtyping._decorator as dec
 original=dec._check_dataclass_annotations
 from native_guide_reuse_v1.typecheck import install as fast_install
 fast=fast_install();assert fast is not None
 packet=dict(image={k:torch.arange(192,dtype=torch.float32).reshape(1,3,8,8)/256 for k in('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},image_mask={k:torch.ones(1,dtype=torch.bool)for k in('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')},state=torch.zeros(1,32),tokenized_prompt=torch.zeros(1,48,dtype=torch.int64),tokenized_prompt_mask=torch.ones(1,48,dtype=torch.bool))
 cases=[packet,dict(packet,state=torch.zeros(2,32)),dict(packet,image={k:v.long()for k,v in packet['image'].items()}),{k:v for k,v in packet.items()if k!='tokenized_prompt_mask'}]
 def run(fn,p):
  dec._check_dataclass_annotations=fn;before=torch.get_rng_state().clone()
  try:r=('ok',Observation.from_dict(copy.deepcopy(p)))
  except Exception as e:r=(type(e).__name__,str(e))
  assert torch.equal(before,torch.get_rng_state());return r
 for item in cases:
  a,b=run(original,item),run(fast,item);assert a[0]==b[0]
  if a[0]=='ok':
   assert all(torch.equal(getattr(a[1],k),getattr(b[1],k))for k in('state','tokenized_prompt','tokenized_prompt_mask'))
   assert all(torch.equal(a[1].images[k],b[1].images[k])for k in a[1].images)
  else:assert a[1]==b[1]
 for name in('jax._src.tree_util','flax.nnx.transforms.compilation'):
  ns=dict(__name__=name,Observation=Observation,copy=copy,packet=cases[1]);exec('def call():return Observation.from_dict(copy.deepcopy(packet))',ns)
  for fn in(original,fast):dec._check_dataclass_annotations=fn;ns['call']()
 dec._check_dataclass_annotations=fast;assert not torch.cuda.is_initialized();return True
