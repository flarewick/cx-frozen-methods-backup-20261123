"""Independent original classes; the only executable edit is official state storage."""
import inspect,textwrap,ast
from native_object_feedback_v3 import tracking,source
text=textwrap.dedent(inspect.getsource(tracking.RealObjectStream))
assert text.count('offload_state_to_cpu=False')==1
candidate=text.replace('offload_state_to_cpu=False','offload_state_to_cpu=True')
assert ast.dump(ast.parse(candidate.replace('offload_state_to_cpu=True','offload_state_to_cpu=False')))==ast.dump(ast.parse(text))
namespace=dict(tracking.__dict__)
exec(compile(candidate,__file__,'exec'),namespace)
CPUStateStream=namespace['RealObjectStream']
namespace=dict(tracking.__dict__,RealObjectStream=CPUStateStream)
exec(compile(textwrap.dedent(inspect.getsource(tracking.MaskCore)),__file__,'exec'),namespace)
CPUStateCore=namespace['MaskCore']
namespace=dict(source.__dict__,MaskCore=CPUStateCore)
exec(compile(textwrap.dedent(inspect.getsource(source.SourceStream)),__file__,'exec'),namespace)
CPUStateSource=namespace['SourceStream']
