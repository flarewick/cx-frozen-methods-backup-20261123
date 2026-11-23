import types
def original_controls(execute,precision_code):
 def walk(code):
  yield code
  for value in code.co_consts:
   if isinstance(value,types.CodeType):yield from walk(value)
 expected={'execute.<locals>.run_case.<locals>.diagnostic_owner','execute.<locals>.run_case.<locals>.tracker.<locals>.operation'}
 found=[code for code in walk(execute.__code__)if code.co_qualname in expected]
 assert len(found)==2 and {code.co_qualname for code in found}==expected
 assert precision_code.co_name=='check'and precision_code.co_qualname=='install.<locals>.check'
 return tuple(found)+(precision_code,)
