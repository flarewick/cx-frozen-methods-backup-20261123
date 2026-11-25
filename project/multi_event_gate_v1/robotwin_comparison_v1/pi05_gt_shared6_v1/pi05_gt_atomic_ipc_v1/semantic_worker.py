from .atomic import install
install()
if __name__=='__main__':
 import runpy
 runpy.run_module('pi05_gt_paired_v1.semantic_worker',run_name='__main__')
