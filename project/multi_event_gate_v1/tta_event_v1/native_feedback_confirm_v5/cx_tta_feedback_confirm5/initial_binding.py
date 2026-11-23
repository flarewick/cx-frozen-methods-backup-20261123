"""Bind the actual first policy observation to the prepared same-scene raw q0."""
import numpy as np
from .paths import digest
def verify_initial(current,item):
 file=item['initial_observation_file']
 assert digest(file)==item['initial_observation_sha256']
 with np.load(file,allow_pickle=False)as z:
  assert all(np.array_equal(current['rgb'][i],z['cam'+str(i)])for i in range(3)),'Actual q0 RGB differs from CPU-preparation observation'
  assert np.array_equal(current['command_state'],z['command_state']),'Actual q0 command state differs'
 return True
