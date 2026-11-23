"""Actual data gate must reject the unfinished collection before any fit."""
from pathlib import Path
import torch
from .data import load_all
from .train import O
from native_policy_execution_v1.common import write,sha
def main():
 assert not torch.cuda.is_initialized()and not(O/'runs').exists()
 try:load_all()
 except ValueError as error:assert str(error)=='Full325 qualified mixed dataset absent: model and optimizer forbidden'
 else:raise AssertionError('Unfinished full collection released fitting')
 assert not(O/'runs').exists()and not torch.cuda.is_initialized()
 write(O/'CPU_actual_missing_data_gate_v22.json',dict(passed=True,no_model_or_optimizer_created=True,optimizer_steps=0,unfinished_collection_refused=True,CUDA_initialized=False,source_sha256={str(p):sha(p)for p in Path(__file__).resolve().parent.glob('*.py')}));print('ACTUAL_MIXED_FULL250_GATE_REFUSED_BEFORE_MODEL_OR_OPTIMIZER')
if __name__=='__main__':main()
