"""Complete fixed16 TRAIN replay with crossed image/state observations."""
import argparse
import atexit
import importlib
import os
from pathlib import Path
import signal
import time
import traceback
import uuid
import numpy as np
from .paths import PROJECT,OUTPUT,read,write,output,verify,digest
from .contract import BRANCHES,dataset,load_journal,commit,validate_packet
from .views import VIEWS,view_inputs,batched_encode
from cx_tta_clock.clock import ACTIVE,Clock,ClockComplete,TICKS,instrument,passive_observation
from cx_tta_clock.runtime import rng_components
from .throughput import TaskPool,replay_prefix,discard_observation,assert_original_parity


class Observer:
    def __init__(self,task,encoder,projection,instruction):
        self.task=task;self.projection=projection;self.instruction=instruction;self.anchor=None;self.encoder_seconds=0.;self.observation_seconds=0.;self.latest_rgb=None
        def copied_encoder(views,prompt):
            from .rgb import capture_views
            self.latest_rgb=capture_views(views)
            return encoder(views,prompt)
        self.encoder=copied_encoder

    def physical(self):
        from cx_tta.attribution_physics import physical_snapshot
        from cx_tta.eval_diagnostics import json_hash
        value=physical_snapshot(self.task)
        if not value['usable']:raise ValueError('Incomplete readable physical state')
        return json_hash(value)

    def read(self):
        from cx_tta.deploy import unpack_observation
        from cx_tta.structured_features import project
        from cx_tta.eval_diagnostics import array_hash
        from cx_tta.recovery_contract import rng_capture,rng_fingerprint
        before=self.physical()
        def extract():
            began=time.monotonic();obs=self.task.get_obs();images,state=unpack_observation(obs);self.observation_seconds+=time.monotonic()-began
            if self.anchor is None:self.anchor=([im.copy() for im in images],state.copy())
            views=view_inputs(images,state,*self.anchor)
            rng=rng_fingerprint(rng_capture());began=time.monotonic();joint,meta=self.encoder(views,self.instruction);self.encoder_seconds+=time.monotonic()-began
            if rng!=rng_fingerprint(rng_capture()):raise ValueError('Controlled encoder consumed RNG')
            visual,_=project(joint,self.projection)
            hashes=lambda rgb:{str(i):array_hash(im) for i,im in enumerate(rgb)}
            return dict(joint=joint.copy(),visual=visual.copy(),command_state=state.copy(),
                images=hashes(images),view_images=[hashes(rgb) for rgb,_ in views],**meta)
        result=passive_observation(self.task,extract)
        if self.physical()!=before:raise ValueError('Controlled observer altered physics')
        result['rgb']=tuple(im.copy() for im in self.latest_rgb)
        return result

