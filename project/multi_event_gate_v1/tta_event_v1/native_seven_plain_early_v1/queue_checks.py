import json
from .common import T

def main():
 import torch
 torch.set_num_threads(2)
 from .branch_checks import check
 result=check();assert not torch.cuda.is_initialized()
 out=T/'outputs/native_seven_plain_early_v1';out.mkdir(exist_ok=True)
 file=out/'CPU_queues_v1.json';assert not file.exists();file.write_text(json.dumps(dict(passed=True,checks=result,CUDA_initialized=False,not_formal_launch_qualification=True),indent=2)+'\n')
 print('FORMAL_QUEUE_CONTROLS',len(result))
if __name__=='__main__':main()
