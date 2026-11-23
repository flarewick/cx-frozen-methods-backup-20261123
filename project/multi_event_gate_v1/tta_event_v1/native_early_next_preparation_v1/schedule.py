"""Independent bookkeeping for the already required next fixed expert inputs."""
import ast
import copy
import inspect
import threading
import textwrap
import types


def ready(work, spec, original_ready):
    if original_ready.is_set():return True
    if not spec.get('future_preparation_population'):return False
    assert spec['future_preparation_population']==dict(split='fit',seed_start=7124000,first_index=0)
    assert spec['dataset_split']=='adapter_validation'
    with work.condition:return len(work.ready)>=9


def isolated(schedule, raw_next, lane_factory):
    private=copy.copy(schedule)
    private.state=copy.deepcopy(schedule.state)
    private.tasks={}
    private.lanes={task:lane_factory('fixed_next_expert_'+task)for task in schedule.spec['tasks']}
    private.locks={task:threading.Lock()for task in schedule.spec['tasks']}
    private.warmed=set()
    private.q0_pending={}
    private.on_initial=None
    # The original base screen closes its expert/q0 scenes. There is no policy
    # handoff consumer for raw next-population inputs, so do not invoke detach.
    private.original_next=types.MethodType(raw_next,private)
    def next_case(name):
        with private.serial_screen_lock.screen(private.allow_future):
            return private.original_next(name)
    private.next=next_case
    return private


def build_prepare(original):
    """Only the cross-population path can bypass a parent-completion read."""
    node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    before=copy.deepcopy(node)
    current=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name)and t.id=='current'for t in n.targets))
    assertion=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assert)
                   and ast.unparse(n.test)=="all((v['accepted'] == end for v in current.values()))")
    assert assertion==current+2 and ast.unparse(node.body[current+1])=="end = spec['episodes_per_task']"
    original_read=copy.deepcopy(node.body[current]);original_assert=copy.deepcopy(node.body[assertion])
    conditional=ast.parse("if not spec.get('early_fixed_next_population_preparation'):\n current=read(root/'screening_complete.json')['state']\n assert all(v['accepted']==end for v in current.values())\nelse:\n assert spec['future_preparation_population']==dict(split='fit',seed_start=7124000,first_index=0)\n assert spec['dataset_split']=='adapter_validation'\n current=None").body[0]
    node.body[current:assertion+1]=[copy.deepcopy(node.body[current+1]),conditional]
    restored=copy.deepcopy(node)
    restored.body[current:current+2]=[original_read,copy.deepcopy(before.body[current+1]),original_assert]
    assert ast.dump(before)==ast.dump(restored),'Original preparation operations changed'
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
    return namespace[node.name]


def adopt_closed_tasks(schedule, private):
    # Called on the same actual owner thread before its original full cleanup.
    assert threading.get_ident()==schedule.c.service.owner
    for name,task in private.tasks.items():
        key='fixed_next_closed_'+name
        assert key not in schedule.tasks
        schedule.tasks[key]=task
    private.tasks.clear()
