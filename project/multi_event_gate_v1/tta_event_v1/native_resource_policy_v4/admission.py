"""Prospective user-authorized admission only; all runtime guards remain intact."""
import json
from pathlib import Path
import sys
T=Path(__file__).resolve().parents[1];P=T.parents[1]
sys.path.insert(0,str(P/'deployment'))
import gpu_doctor
from native_fast_inventory_v1.inventory import inventory
POLICY=json.loads((Path(__file__).parent/'policy.json').read_text())
assert POLICY['maximum_account_GPUs_all_hours']==3 and POLICY['existing_card_75percent_or_50percent_memory_gate']is False

def require_admission(fleet,uuid):
 cards=fleet['cards'];selected=next((card for card in cards if card['uuid']==uuid),None)
 if selected is None:raise RuntimeError('Selected GPU is not present in fresh inventory')
 if not selected['available']or selected['pids']or selected['memory_mib']>64 or selected['sm']>5:
  raise RuntimeError('Selected GPU is not genuinely empty; preserve existing jobs')
 count=sum(bool(card['own_pids'])for card in cards)
 if fleet['own_gpu_count']!=count:raise RuntimeError('Inconsistent account GPU count')
 if count>=3:raise RuntimeError('All-hours account GPU limit is3; refusing another GPU')
 return selected

def check_admission(info):
 gpu_doctor.verify_server_rules();fleet=inventory();require_admission(fleet,info['uuid'])
 print('USER_EMPTY_CARD_UNDER3_ADMISSION',json.dumps(dict(policy_revision=POLICY['revision'],fleet=fleet,runtime_guards_unchanged=True)),flush=True)
 return fleet
