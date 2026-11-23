"""Object-local command support, explicit free dynamics, original radial/linear kernel."""
import numpy as np
ARMS=[]
for arm in range(2):
 ids=[]
 for first in(0,16,40,56):
  ids.extend((first+np.arange(16).reshape(4,2,2)[:,arm]).ravel().tolist())
 for first in(72,120):
  ids.extend((first+np.arange(48).reshape(4,2,6)[:,arm]).ravel().tolist())
 ids.extend((168+np.arange(24).reshape(4,2,3)[:,arm]).ravel().tolist())
 ids.extend([32+arm,34+arm,36+arm]);ARMS.append(np.array(sorted(ids),int))
assert set(ARMS[0]).isdisjoint(ARMS[1])and set(np.r_[*ARMS])==set(range(192))-{38,39}
def support(x,u,points):
 x=np.asarray(x,float);u=np.asarray(u,float);points=np.asarray(points,float)
 if x.shape[1]!=53 or u.shape[1]!=197 or points.shape!=(len(x),24,2):raise ValueError('Exact legal native geometry/command schema')
 tcp=x[:,19:23].reshape(-1,2,2)*320
 path=np.concatenate([tcp[:,None],tcp[:,None]+u[:,:16].reshape(-1,4,2,2)*320],axis=1)
 delta=path[:,:,:,None]-points[:,None,None];squared=np.sum(delta*delta,axis=-1)
 squared=np.where(np.isfinite(squared),squared,np.inf);distance=np.sqrt(squared.min(axis=(1,3)))
 radius=np.maximum(x[:,27]*320,1.)
 return np.exp(-.5*(distance/radius[:,None])**2)
def coordinates(x,u,points,mode):
 x=np.asarray(x,float).copy();u=np.asarray(u,float)
 if mode=='without_history':x[:,34:44]=0
 v=u[:,:192].copy();v[:,[32,33,36,37]]-=np.clip(x[:,[6,13,6,13]]*np.pi,0,1)/.1
 gate=support(x,u,points)
 if mode=='without_locality':gate[:]=1.
 for arm,ids in enumerate(ARMS):v[:,ids]*=gate[:,arm,None]
 v[:,38:40]=0
 if mode=='history_only':v[:]=0
 if mode not in('correct','wrong_command','without_locality','without_history','history_only'):raise ValueError('Unregistered fixed mechanism control')
 v=np.c_[v,x[:,34:36]*(u[:,38]*20)[:,None]]
 assert np.isfinite(v).all()
 return x,v,u[:,-5:].copy(),gate
def state(x,v,kind):
 scale=np.maximum(x.std(0),.05);scale[34:42]=np.maximum(x[:,34:42].std(0),.005);scale[[43,52]]=1.
 return dict(context_mean=x.mean(0),context_scale=scale,action_scale=np.maximum(np.sqrt(np.mean(v*v,0)),.005),
  train_context=x.copy(),train_action=v.copy(),train_kind=kind.copy())
def kernel(s,x,v,kind):
 left=(x-s['context_mean'])/s['context_scale'];right=(s['train_context']-s['context_mean'])/s['context_scale']
 distance=np.maximum(np.mean(left*left,1)[:,None]+np.mean(right*right,1)[None]-2*left@right.T/left.shape[1],0)
 context=np.exp(-.25*distance);phase=.2+.8*(kind@s['train_kind'].T)
 dot=(v/s['action_scale'])@(s['train_action']/s['action_scale']).T/v.shape[1]
 return context*phase*dot
def predict(s,x,v,kind):return kernel(s,x,v,kind)@s['weights']
