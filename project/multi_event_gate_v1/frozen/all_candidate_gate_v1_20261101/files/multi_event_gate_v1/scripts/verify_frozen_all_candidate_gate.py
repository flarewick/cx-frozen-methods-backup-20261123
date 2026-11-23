"""CPU-only, standard-library content check. Does not import policy, CUDA, or a launcher."""
import argparse,hashlib,json,sys
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 w=Path(__file__).resolve().parents[1]
 ap=argparse.ArgumentParser(__doc__)
 ap.add_argument('--manifest',type=Path,default=w/'frozen/all_candidate_gate_v1_20261101/manifest.json')
 ap.add_argument('--scope',choices=('algorithm','all'),default='all')
 ap.add_argument('--archive-only',action='store_true',help='Check archived bytes only; do not claim the live working tree matches.')
 a=ap.parse_args();path=a.manifest.resolve();pin=path.with_name('manifest.sha256')
 expected=pin.read_text().split()[0]
 if digest(path)!=expected:raise SystemExit('FAIL: manifest differs from manifest.sha256')
 m=json.loads(path.read_text());failed=[];checked=0
 for row in m['files']:
  if a.scope=='algorithm'and row['role']not in ('algorithm','model'):continue
  snap=path.parent/row['snapshot'];expected=row['sha256']
  if not snap.is_file()or digest(snap)!=expected:failed.append(dict(path=str(snap),reason='snapshot missing or changed'))
  checked+=1
  if not a.archive_only:
   live=Path(row['original_host_path'])
   if not live.is_file()or digest(live)!=expected:failed.append(dict(path=str(live),reason='live file missing or changed'))
   checked+=1
 out=dict(passed=not failed,freeze_id=m['freeze_id'],scope=a.scope,archive_only=a.archive_only,checked_file_hashes=checked,failed=failed,CPU_only=True,no_GPU_or_model_import=True,not_a_runtime_or_GPU_resource_acceptance=True)
 print(json.dumps(out,ensure_ascii=False,indent=2))
 return 1 if failed else 0
if __name__=='__main__':sys.exit(main())
