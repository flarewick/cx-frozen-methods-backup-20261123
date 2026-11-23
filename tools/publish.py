"""Create the authorized private GitHub backup and verify the pushed commit."""
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPRedirectHandler

OWNER = 'flarewick'
NAME = 'cx-frozen-methods-backup-20261123'
DESCRIPTION = 'Private backup of the frozen event_full, all-candidate gate and TTA methods (2026-11-23)'
ROOT=Path(__file__).resolve().parents[1]
SLUG=f'{OWNER}/{NAME}'
URL=f'https://github.com/{SLUG}.git'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def read_credential():
    for name in ('GH_TOKEN','GITHUB_TOKEN'):
        if os.environ.get(name):
            return os.environ[name]
    try:
        result=subprocess.run(['gh','auth','token','--hostname','github.com'],capture_output=True,text=True,check=True)
        if result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    if not sys.stdin.isatty():
        raise SystemExit('Set GH_TOKEN locally, or run in a terminal for a hidden token prompt.')
    return getpass.getpass('GitHub token (hidden, memory only): ').strip()


def publish():
    local=subprocess.check_output(['git','rev-parse','refs/heads/main'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise SystemExit('Check the local backup commit before uploading.')
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/snapshot.py'),'verify'],cwd=ROOT,check=True)
    token=read_credential()
    if not token:
        raise SystemExit('No GitHub credential available.')
    opener=build_opener(NoRedirect())
    def api(path, data=None):
        req=Request('https://api.github.com/'+path,
                    data=json.dumps(data).encode() if data else None,
                    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                             'Content-Type':'application/json','User-Agent':'cx-frozen-backup'})
        with opener.open(req,timeout=30) as response:
            return json.load(response)
    if api('user')['login'].casefold()!=OWNER:
        raise SystemExit(f'Authenticated GitHub account must be {OWNER}.')
    try:
        repo=api('repos/'+SLUG)
    except HTTPError as error:
        if error.code!=404:
            raise
        repo=api('user/repos',dict(name=NAME,private=True,description=DESCRIPTION,auto_init=False))
    if repo['full_name'].casefold()!=SLUG or not repo['private'] or repo['description']!=DESCRIPTION or not repo.get('permissions',{}).get('push',False):
        raise SystemExit('Existing repository identity, visibility, description, or write permission is unexpected.')
    # Credential is passed only through the subprocess environment to an isolated helper.
    env=dict(os.environ,CX_BACKUP_TOKEN=token,GIT_TERMINAL_PROMPT='0')
    helper='!f() { test "$1" = get && printf "username=x-access-token\\npassword=%s\\n\\n" "$CX_BACKUP_TOKEN"; }; f'
    git=['git','-c','credential.helper=','-c','credential.helper='+helper,
         '-c','credential.interactive=never','-c','http.followRedirects=false']
    def remote_tip():
        result=subprocess.run([*git,'ls-remote',URL,'refs/heads/main'],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
        return result.stdout.split()[0] if result.stdout.strip() else None
    base=remote_tip()
    if base not in (None,local):
        raise SystemExit('Remote main already has a different commit.')
    if base is None:
        subprocess.run([*git,'push',URL,'refs/heads/main:refs/heads/main'],cwd=ROOT,env=env,check=True)
    if remote_tip()!=local:
        raise SystemExit('Remote commit did not match the verified local snapshot.')
    print(json.dumps({'repository':repo['html_url'],'private':True,'main_commit':local,'remote_verified':True}))


if __name__=='__main__':
    try:
        publish()
    except HTTPError as error:
        sys.exit(f'GitHub API returned HTTP {error.code}. Check repository creation and write scopes.')
