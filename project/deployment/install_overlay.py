"""Apply audited source-level patches inside the newly built environment only."""
import hashlib
import json
from pathlib import Path
import shutil
import site


def main():
    overlay=Path('/opt/cx_overlay/site_packages')
    if not overlay.is_dir():raise FileNotFoundError('Missing environment source overlay')
    target=Path(site.getsitepackages()[0])
    for path in overlay.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(overlay);dest=target/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    env=json.loads(Path('/opt/cx_deployment/environment.json').read_text())
    for row in env['modified_package_sources']:
        dest=target/row['path']
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Overlay mismatch: '+str(dest))
    # OpenPI also imports this new (non-wheel) support file when constructing
    # PI0Pytorch. It is not a modified wheel file, so it is absent from the
    # original 15-file diff inventory. Restore the verified original bytes.
    support=Path('/data/cj/robo_pi/250010033/qfw/robotwin/RoboTwin/policy/pi05/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py')
    support_sha='466e8eb7887ac5ccc98118a3169298d6ab65a284d834bae74751e3854dbc4245'
    if hashlib.sha256(support.read_bytes()).hexdigest()!=support_sha:
        raise ValueError('OpenPI transformer support source differs from the transfer manifest')
    support_dest=target/'transformers/models/siglip/check.py'
    shutil.copy2(support,support_dest)
    if hashlib.sha256(support_dest.read_bytes()).hexdigest()!=support_sha:
        raise ValueError('OpenPI transformer support copy mismatch')
    print('Audited package overlays installed:',len(env['modified_package_sources']))
    print('Verified original OpenPI transformer support installed:',support_sha)


if __name__=='__main__':main()
