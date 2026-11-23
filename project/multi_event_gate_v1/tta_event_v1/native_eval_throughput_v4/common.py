from pathlib import Path
import hashlib,json,os
C=Path(__file__).resolve().parent;T=C.parent;P=T.parents[1];COHORT=os.environ.get('CX_ONPOLICY_STAGE','branch12')
PRIMARY=T/'outputs/native_eval_throughput_v4'/COHORT
PREPARED_RECOVERY=os.environ.get('CX_OFFICIAL_PREPARED_RECOVERY','0')=='1'
O=PRIMARY/'prepared_recovery_001'if PREPARED_RECOVERY else PRIMARY
CANON=Path('/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA')
TASKS=('open_laptop','grab_roller','beat_block_hammer','turn_switch','adjust_bottle')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def mapped(p):
 p=Path(p)
 if p.exists():return p
 if p.is_relative_to(CANON):return P/p.relative_to(CANON)
 host=Path('/data/qiufw/cx_migration/cx_new_direction_guide_TTA')
 if p.is_relative_to(host):return P/p.relative_to(host)
 return p

def write(p,v):
 p=Path(p).resolve();assert p.is_relative_to(O.resolve())
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.'+str(os.getpid())+'.publish')
 with tmp.open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 try:os.link(tmp,p)
 finally:tmp.unlink()
def replace(p,v):
 p=Path(p).resolve();assert p.is_relative_to(O.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def verify(spec):
 for field in('source_sha256','input_sha256'):
  for name,digest in spec.get(field,{}).items():
   p=mapped(name)if Path(name).is_absolute()else P/name
   assert sha(p)==digest,('Frozen evaluation dependency changed',str(p))
def batch(n):return O/'batches'/('%03d'%n)/'runs'/os.environ['CX_BEST100_RUN_ID']

# Earlier qualified science is immutable. A failed cold attempt is never a result.
def stage_folder(number):
 assert 0<=number<20
 namespace='native_calibrated_five100_v20'if number<4 else ('native_calibrated_five100_v21'if number==4 else ('native_calibrated_five100_v22'if number<7 else 'native_eval_throughput_v4'))
 folder=T/'outputs'/namespace/('official_'+str(number).zfill(3))
 recovery=folder/'prepared_recovery_001'
 return recovery if recovery.exists()else folder


def segment_folder(number,part):
 assert 7<=number<20 and part in(0,1)
 namespace='native_calibrated_five100_v25'if number<13 else('native_calibrated_five100_v26'if(number,part)==(13,0)else('native_calibrated_five100_v27'if number<16 or(number,part)==(16,0)else('native_calibrated_five100_v28'if number<17 or(number,part)==(17,0)else('native_calibrated_five100_v29'if number<18 or(number,part)==(18,0)else('native_calibrated_five100_v30'if number==18 else('native_calibrated_five100_v32'if part==0 else'native_eval_throughput_v4'))))))
 folder=T/'outputs'/namespace/('official_'+str(number).zfill(3)+'_p'+str(part))
 recovery=folder/'prepared_recovery_001'
 return recovery if recovery.exists()else folder

def decoded_stage(cohort):
 role,number,part=cohort.split('_');number=int(number);part=int(part.removeprefix('p'))
 assert role=='official'and 7<=number<20 and part in(0,1)
 return number,part,3 if part==0 else 2
