"""Official SAM2 CPU state storage; unchanged GPU compute, pending raw bridges.

Only offload_state_to_cpu differs from the frozen Stream initializer. GPU raw
logits/presence must match all previously recorded prefix frames before acceptance.
"""
import numpy as np
from query_object_v1.stream import MaskCore,snapshot,preserved,video

class OffloadedStream(video.Stream):
    def __init__(self,model,initial_rgb,initial_masks):
        import sam2.sam2_video_predictor as official
        if np.asarray(initial_masks).shape!=(2,*initial_rgb.shape[:2]):
            raise ValueError('Two actual observed initial masks required')
        official.load_video_frames=video.arrived_loader
        self.model=model
        self.frames=video.ArrivedFrames(initial_rgb,model.image_size)
        self.state=model.init_state(self.frames,offload_video_to_cpu=True,offload_state_to_cpu=True)
        self.index=0
        for obj,mask in enumerate(initial_masks):model.add_new_mask(self.state,frame_idx=0,obj_id=obj,mask=mask.copy())
        result=list(model.propagate_in_video(self.state,start_frame_idx=0,max_frame_num_to_track=0))
        assert len(result)==1 and result[0][0]==0 and result[0][1]==[0,1]
        self.initial_logits=result[0][2].detach().cpu().numpy().copy()

class OffloadedCore(MaskCore):
    def initialize(self,rgb,masks,*,tick,decision_tick):
        import hashlib
        if tick!=0 or decision_tick!=tick or self.frames:
            raise ValueError('First actually observed image only')
        before=snapshot(self.model)
        self.stream=OffloadedStream(self.model,rgb,masks)
        self.frames=self.stream.frames
        logits=self.stream.initial_logits
        score=np.array([float(self.stream.state['output_dict_per_obj'][obj]['cond_frame_outputs'][0]['object_score_logits'].sigmoid().cpu().item())for obj in range(2)])
        preserved(self.model,before)
        return dict(tick=0,decision_tick=0,masks=(logits[:,0]>0),presence=score,
                    raw_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest())
