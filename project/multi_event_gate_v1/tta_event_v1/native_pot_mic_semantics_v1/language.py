"""Exact language-only single-object association for pot and microphone transfer."""
import re

VERSION = 'native_pot_mic_semantics_v1'
PATTERNS = {'pot': r'\b(?:kitchenpots?|pots?)\b', 'microphone': r'\b(?:microphones?|mics?)\b'}
OTHER_OBJECTS = r'\b(?:bottles?|bowls?|cups?|hammers?|mallets?|blocks?|cubes?|rollers?|laptops?|notebooks?|shoes?|bread|staplers?|fans?|mats?|pads?)\b'

def new_prompt(language):
    if not isinstance(language, str):
        raise ValueError('Actual instruction text required')
    text = language.lower()
    matches = {name: list(re.finditer(pattern, text)) for name, pattern in PATTERNS.items()}
    names = [name for name, found in matches.items() if found]
    missing = dict(directed=False, target=None, reference=None, kind='unsupported_or_ambiguous_language', interface_version=VERSION)
    if len(names) != 1 or re.search(OTHER_OBJECTS, text):
        return missing, []
    name = names[0]
    # A named switch beside a microphone is not a microphone handover.
    if name == 'microphone' and re.search(r'\b(?:press|click|toggle|turn)\b', text):
        return missing, []
    match = matches[name][0]
    start = match.start()
    determiners = list(re.finditer(r'\b(?:the|a|an|this|that)\s+', text[:start]))
    if determiners:
        candidate = text[determiners[-1].end():start]
        if re.fullmatch(r'(?:[a-z]+(?:-[a-z]+)*\s+){0,6}', candidate) and not re.search(r'\b(?:and|or|on|in|into|with|from|to|of|use|using|grab|pick|lift|raise|place|turn|hold|pass|hand|transfer|give)\b', candidate):
            start = determiners[-1].end()
    phrase = text[start:match.end()]
    assert phrase and phrase in text
    return dict(directed=False, target=name, reference=None, kind='explicit_single_object', interface_version=VERSION, actual_object_count=1, absent_reference_explicit=True, grounding_phrases_exact_instruction_substrings=True), [phrase]

def prompt(language):
    roles, phrases = new_prompt(language)
    if phrases or any(re.search(pattern, language.lower()) for pattern in PATTERNS.values()):
        return roles, phrases
    from native_object_association_v1.semantic import prompt as previous
    return previous(language)
