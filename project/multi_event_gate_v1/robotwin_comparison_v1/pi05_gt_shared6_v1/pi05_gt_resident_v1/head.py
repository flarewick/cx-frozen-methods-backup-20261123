from pathlib import Path
import json,torch
from .common import C,W,sha
from cx_tta.stacking_adapter import StackingAdapter
from cx_tta.robot_kinematics import RobotFK
SOURCE_SHA='6e0de463a03d8bac0e160aa46cf955f8575b78ecd38d1e799392e253a7d289e7'
def build_head(training):
 training=Path(training).resolve();assert training== (C/'pi05_gt_shared6_v1/frozen_shared6_head_v1').resolve()
 source=W/'outputs/training_diagnostics/shared6_vs_single_20261025/shared6_lambda0p0/best_scaled.pt';assert sha(source)==SOURCE_SHA
 current=torch.load(source,map_location='cpu',weights_only=True);payload=torch.load(training/'adapter_seed0.pt',map_location='cpu',weights_only=True)
 assert set(payload['model'])==set(current['model'])and all(torch.equal(v,payload['model'][k])for k,v in current['model'].items())
 meta=payload['metadata'];model=StackingAdapter(32,meta['geometry_mean']);model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
 robot=json.loads(Path(meta['robot_config_path']).read_text());return model,[RobotFK(robot,s)for s in ['left','right']]
