from native_object_feedback_v4.language import new_prompt as descriptive
VERSION='native_object_feedback_v5'
def new_prompt(language):
 roles,phrases=descriptive(language);return dict(roles,interface_version=VERSION),phrases
def prompt(language):
 from native_transfer_semantics_v2.semantic import prompt as previous
 roles,phrases=previous(language)
 return(roles,phrases)if phrases else new_prompt(language)
