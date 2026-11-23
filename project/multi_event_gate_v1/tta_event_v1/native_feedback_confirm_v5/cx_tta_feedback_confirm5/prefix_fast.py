"""Original native CPU control only while the measurement clock is inactive.

The timed500step intervention/observer path never uses this fast loop. Every
native prefix render/success callback remains, including early-stop behavior.
"""
import ast,inspect,types
from pathlib import Path
from .paths import OUTPUT,read


def original_function(source,namespace,filename):
    tree=ast.parse(source)
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='take_action']
    if len(nodes)!=1 or nodes[0].decorator_list:raise ValueError('Need one original undecorated native take_action')
    node=nodes[0]
    def find(code):
        for value in code.co_consts:
            if isinstance(value,types.CodeType):
                if value.co_name=='take_action':return value
                result=find(value)
                if result is not None:return result
        return None
    # Compile the WHOLE immutable module but never execute it. Python compiles
    # imported-global calls differently when a method is lifted out of its
    # module/class context; retain the exact original function code object.
    code=find(compile(source,filename,'exec'))
    if code is None or code.co_freevars:raise ValueError('Unexpected native action closure')
    defaults=tuple(ast.literal_eval(v) for v in node.args.defaults)
    return types.FunctionType(code,namespace,'take_action',defaults)



def dispatch(original,fast):
    from cx_tta_clock.clock import ACTIVE
    def call(action,action_type='qpos'):
        if ACTIVE.get() is not None:return original(action,action_type)
        return fast(action,action_type)
    return call


def install(task,gate):
    if getattr(task,'_cx_controlled_prefix_fast',False):return
    import native_generic_physics as helper
    spec=read(OUTPUT/'protocol.json');file=Path(spec['runtime_source'])/'envs/_base_task.py'
    instrumented=task.take_action
    original=inspect.unwrap(instrumented.__func__)
    fn=original_function(file.read_text(),original.__globals__,str(file))
    # The helper independently checks the immutable ORIGINAL source and its
    # existing CPU-SAPIEN/AST/rounding/binary proof. No shared source is patched.
    task.take_action=types.MethodType(fn,task)
    try:helper.install_task(task,gate,helper.load_native());fast=task.take_action
    except BaseException:
        task.take_action=instrumented;raise
    task.take_action=dispatch(instrumented,fast)
    task._cx_controlled_prefix_fast=True
