"""Exact determiner-delimited nominal modifiers, including hyphenated words."""
import re
from native_object_feedback_v3.language import new_prompt as noun
from native_object_feedback_v4.language import PATTERNS
VERSION='native_object_association_v1'
def prompt(language):
 roles,phrases=noun(language)
 if not phrases:return dict(roles,interface_version=VERSION),[]
 text=language.lower();out=[]
 for name in [roles['target']]+([roles['reference']]if roles['reference']else[]):
  match=re.search(PATTERNS[name],text);assert match
  prefix=text[:match.start()];determiners=list(re.finditer(r'\b(?:the|a|an|this|that)\s+',prefix))
  start=match.start()
  if determiners:
   candidate=prefix[determiners[-1].end():]
   # A determiner may also belong to an earlier noun phrase. Only an
   # uninterrupted nominal modifier span is admitted; no verbs/relations.
   if re.fullmatch(r'(?:[a-z]+(?:-[a-z]+)*\s+){0,5}',candidate) and not re.search(r'\b(?:and|or|on|in|into|with|from|to|of|using|use|grab|pick|lift|place|turn|open|adjust|move|put|set|hit|strike|hold|press)\b',candidate):start=determiners[-1].end()
  phrase=text[start:match.end()];assert phrase in text;out.append(phrase)
 return dict(roles,interface_version=VERSION,grounding_phrases_exact_instruction_substrings=True),out
