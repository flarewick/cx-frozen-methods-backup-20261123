"""Pending work and available CPU threads only; no model features or labels."""
def allocate(remaining,pending,free_workers):
 assert 0<=remaining<=8 and 0<=pending and 0<=free_workers<=4
 if not pending or not free_workers:return 0
 target=remaining//min(pending,free_workers)
 return next((value for value in(8,4,2)if value<=target),0)
