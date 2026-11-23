"""Exact words from the real instruction, with explicit descriptive support."""
import re
from native_object_feedback_v3.language import new_prompt as prior
VERSION='native_object_feedback_v4'
PATTERNS={'hammer':r'\b(?:hammer|mallet)s?\b','block':r'\b(?:block|cube)s?\b','switch':r'\b(?:switch(?:es)?|button)s?\b','bottle':r'\bbottles?\b','roller':r'\b(?:roller|rolling\s+pin)s?\b','laptop':r'\b(?:laptop|notebook\s+computer|computer)s?\b'}
ADJECTIVES=set('wooden wood beige black white brown red blue green yellow gray grey foldable curved small angular rectangular silver metal metallic plastic pink purple rubber toggle light dark tiny large long short cylindrical flat thick thin'.split())
def new_prompt(language):
 roles,phrases=prior(language)
 if not phrases:return dict(roles,interface_version=VERSION),[]
 text=language.lower();names=[roles['target']]+([roles['reference']]if roles['reference']else[]);out=[]
 for name in names:
  match=re.search(PATTERNS[name],text);assert match
  start=match.start();tokens=list(re.finditer(r"[a-z]+(?:-[a-z]+)*",text[:start]))
  end=start
  for token in reversed(tokens[-4:]):
   if text[token.end():end].strip()or token.group()not in ADJECTIVES:break
   start=token.start();end=start
  phrase=text[start:match.end()];assert phrase in text;out.append(phrase)
 return dict(roles,interface_version=VERSION,grounding_phrases_exact_instruction_substrings=True),out

def prompt(language):
 from native_transfer_semantics_v2.semantic import prompt as previous
 roles,phrases=previous(language)
 return(roles,phrases)if phrases else new_prompt(language)
