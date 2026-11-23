"""Bounded reconstruction of one fixed old initial state, before any policy action."""
def replay(setup,close,reseed,measure,expected,unstable,*,seed,index,kwargs,note=lambda **kw:None,attempts=5):
 if not isinstance(expected,str)or len(expected)!=64 or not 1<=attempts<=5:raise ValueError('Exact old physical hash and bounded attempts required')
 last=None
 for attempt in range(1,attempts+1):
  reseed(seed)
  try:
   result=setup(now_ep_num=index,seed=seed,**kwargs);actual=measure()
   if actual==expected:
    note(seed=seed,index=index,attempt=attempt,matched=True,actual=actual,expected=expected,policy_actions=0);return result
   last=dict(kind='physical_hash_mismatch',actual=actual,expected=expected)
  except unstable as error:last=dict(kind='unstable_initialization',error=str(error))
  note(seed=seed,index=index,attempt=attempt,matched=False,policy_actions=0,**last)
  close()
 raise RuntimeError('Fixed old initial state not reconstructed before policy: '+str(dict(seed=seed,index=index,attempts=attempts,last=last)))
