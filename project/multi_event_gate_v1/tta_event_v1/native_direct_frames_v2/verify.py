from pathlib import Path
from native_policy_execution_v1.common import T,read,sha,write
from native_onpolicy_recovery_v3.common import mapped
ROOT=T.parents[1]/'deployment/.work/rootfs'
ENV_PREFIXES=('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/','/opt/cx_venv/')

def dependency(name):
 p=Path(name)
 if p.exists():return p
 q=mapped(p)if p.is_absolute()else T.parents[1]/p
 if q.exists():return q
 assert p.is_absolute()and any(str(p).startswith(v)for v in ENV_PREFIXES),('Unsupported missing dependency',name)
 q=ROOT/str(p).lstrip('/');assert q.is_file(),('Missing readonly runtime dependency',str(q))
 return q

def verify_proof(value):
 for field in ('source_sha256','input_sha256'):
  for name,digest in value.get(field,{}).items():assert sha(dependency(name))==digest,('Original dependency hash changed',name)
 return True

def main():
 proof=T/'outputs/native_direct_frames_v1/CPU_contract_checks_v1.json';value=read(proof);assert value['passed']and verify_proof(value)
 actual=T/'outputs/native_direct_frames_v1/CPU_actual_host_exit_v1.json';a=read(actual);assert a['exit_code']==0 and a['actual_Popen_wait_return']and a['proc_absent_after_wait']
 import copy
 wrong=copy.deepcopy(value);name=next(n for n in value['source_sha256']if '/openpi/'in n);wrong['source_sha256'][name]='0'*64
 try:verify_proof(wrong)
 except AssertionError:pass
 else:raise AssertionError('Changed installed environment source accepted')
 C=Path(__file__).parent;write(T/'outputs/native_direct_frames_v2/CPU_source_mapping_checks_v1.json',dict(passed=True,CPU_only=True,all_original_source_hashes_preserved=True,real_installed_environment_hash_mismatch_refused=True,no_copy_symlink_install_or_environment_mutation=True,actual_resolutions={name:str(dependency(name))for name in value['source_sha256']},source_sha256={str(p):sha(p)for p in C.glob('*')if p.is_file()},input_sha256={str(p):sha(p)for p in[proof,actual]}))
 print('READONLY_HOST_CONTAINER_SOURCE_MAPPING_ALL_ORIGINAL_HASHES_PASS',flush=True)
if __name__=='__main__':main()
