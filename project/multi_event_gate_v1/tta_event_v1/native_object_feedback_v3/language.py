"""Language-only object roles; no task ID, simulator state, or language rewriting."""
import re
VERSION='native_object_feedback_v3'
def new_prompt(language):
 text=language.lower()
 patterns={'hammer':r'\b(?:hammer|mallet)s?\b','block':r'\b(?:block|cube)s?\b','switch':r'\b(?:switch(?:es)?|button)s?\b','bottle':r'\bbottles?\b','roller':r'\b(?:roller|rolling\s+pin)s?\b','laptop':r'\b(?:laptop|notebook\s+computer)s?\b'}
 found={name for name,pat in patterns.items()if re.search(pat,text)}
 if 'laptop'not in found and re.search(r'\bcomputer\b',text)and re.search(r'\b(screen|lid)\b',text):found.add('laptop')
 objects=found-{'block'}
 if len(objects)!=1:return dict(directed=False,target=None,reference=None,kind='unsupported_or_ambiguous_language',interface_version=VERSION),[]
 obj=next(iter(objects));reference='block'if obj=='hammer'and'block'in found else None
 if 'block'in found and obj!='hammer':return dict(directed=False,target=None,reference=None,kind='unsupported_or_ambiguous_language',interface_version=VERSION),[]
 phrase='computer'if obj=='laptop'and not re.search(patterns['laptop'],text)else obj
 phrases=[phrase]+([reference]if reference else[])
 return dict(directed=reference is not None,target=obj,reference=reference,kind='explicit_tool_target_pair'if reference else'explicit_single_object',interface_version=VERSION,actual_object_count=len(phrases),absent_reference_explicit=reference is None),phrases

def prompt(language):
 from native_transfer_semantics_v2.semantic import prompt as previous
 roles,phrases=previous(language)
 return (roles,phrases)if phrases else new_prompt(language)
