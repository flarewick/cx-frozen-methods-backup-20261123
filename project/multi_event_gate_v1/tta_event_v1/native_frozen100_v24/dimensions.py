from .common import O,read,TASKS

def layout(study,index):return study['mixed_task_slots'][str(index)]
def lanes(index):return len(layout(read(O/'study.json'),index))
