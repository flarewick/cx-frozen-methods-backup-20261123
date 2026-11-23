"""Default8; explicit permission is required for a declared exploratory7 fit."""
from pathlib import Path
from native_policy_execution_v1.common import T,read,sha
from native_onpolicy_recovery_v3.common import verify
AUTH=T/'outputs/native_fit_support_review_v1/user_support7_authorization.json'
def minimum_fit():
 if not AUTH.exists():return 8
 value=read(AUTH);verify(value)
 assert value['explicit_user_authorization']and value['previous_minimum']==8 and value['approved_minimum']==7
 assert value['scope']=='fixed325_minimum_FIT_support7_exploratory_training_only'
 assert value['fixed1000_and_VAL_and_independent_native_gates_unchanged']
 return 7
def evidence():
 values={str(Path(__file__)):sha(Path(__file__))}
 if minimum_fit()==7:values[str(AUTH)]=sha(AUTH)
 return values
