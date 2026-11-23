"""Same frame objects and original checks without unused source/stack formatting."""
import ast,copy,inspect,linecache,sys,textwrap,types

def frame_infos():
    frame=sys._getframe(1)
    try:
        while frame is not None:
            yield types.SimpleNamespace(frame=frame)
            frame=frame.f_back
    finally:del frame

def rewritten(original,old,new,additions):
    node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    assert isinstance(node,ast.FunctionDef)and not node.decorator_list
    before=copy.deepcopy(node);hits=[]
    class Change(ast.NodeTransformer):
        def visit_Call(self,item):
            if ast.unparse(item)==old:
                hits.append(item);return ast.parse(new,mode='eval').body
            return self.generic_visit(item)
        def visit_Subscript(self,item):
            if ast.unparse(item)==old:
                hits.append(item);return ast.parse(new,mode='eval').body
            return self.generic_visit(item)
    changed=Change().visit(node);assert len(hits)==1,(original.__name__,len(hits))
    restored=copy.deepcopy(changed)
    class Undo(ast.NodeTransformer):
        def visit_Call(self,item):
            if ast.unparse(item)==new:return copy.deepcopy(hits[0])
            return self.generic_visit(item)
    assert ast.dump(Undo().visit(restored))==ast.dump(before),'Original check/options operations changed'
    source=ast.unparse(ast.fix_missing_locations(changed))+'\n';filename='<native_direct_frames_v1:'+original.__name__+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(original.__globals__,**additions)
    exec(compile(source,filename,'exec'),namespace)
    return namespace[original.__name__]

def candidates():
    import warp as wp
    import warp._src.context as context
    from openpi.shared import array_typing
    typing=rewritten(array_typing._check_dataclass_annotations,'inspect.stack()','_original_frame_infos()',dict(_original_frame_infos=frame_infos))
    warp=rewritten(context.set_module_options,'inspect.stack()[1][0]','_original_caller_frame(1)',dict(_original_caller_frame=sys._getframe))
    assert wp.set_module_options is context.set_module_options
    return typing,warp

def install():
    import warp as wp
    import warp._src.context as context
    import jaxtyping._decorator as decorator
    from openpi.shared import array_typing
    assert decorator._check_dataclass_annotations is array_typing._check_dataclass_annotations
    typing,warp=candidates()
    array_typing._check_dataclass_annotations=typing
    decorator._check_dataclass_annotations=typing
    wp.set_module_options=warp;context.set_module_options=warp
