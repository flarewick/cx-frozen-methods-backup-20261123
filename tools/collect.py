"""Build a checksum-preserving snapshot from identified frozen execution records."""
import argparse
import ast
from collections import Counter, defaultdict, deque
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re
import shutil
import subprocess

W = Path('multi_event_gate_v1')
T = W / 'tta_event_v1'
GATE = W / 'frozen/all_candidate_gate_v1_20261101'
TTA = T / 'frozen_tta093_v1'
D = Path('deployment')
PROJECT_PREFIXES = (
    '/data/cj/robo_pi/250010033/VLA-Adapter/cx_new_direction_guide_TTA/',
    '/data/qiufw/cx_migration/cx_new_direction_guide_TTA/',
    '/home/qiufw/cx_migration/cx_new_direction_guide_TTA/',
)
HASH = re.compile(r'^[0-9a-f]{64}$')
SECRET = re.compile(rb'(?<![\w])(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16}|-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|https?://[^/\s:@]+:[^/\s@]+@)', re.I)
TEXT = {'.py', '.sh', '.md', '.json', '.jsonl', '.log', '.txt', '.yml', '.yaml', '.toml', '.cpp', '.h', '.xml', '.urdf', '.srdf', '.sha256', '.cfg', '.ini'}
SOURCE = {'.py', '.sh', '.cpp', '.h', '.hpp', '.cu', '.c', '.yml', '.yaml', '.toml'}
KEYS = {'source_sha256', 'input_sha256', 'reference_input_sha256', 'frozen_files_sha256',
        'source_hashes', 'input_hashes', 'file_sha256', 'asset_sha256', 'source_files_sha256'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relpath(value, base):
    if not isinstance(value, str) or not value or '\x00' in value:
        return None
    for prefix in (*PROJECT_PREFIXES, str(base) + '/'):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path == Path('.'):
        return None
    if any(p in {'.git', '.ssh', '.codex', '.cache', '.aws', '__pycache__', 'backups'} for p in path.parts):
        return None
    if path.name in {'.env', 'hosts.yml', 'auth.json', 'credentials.json'} or path.suffix in {'.pem', '.key', '.pyc'}:
        return None
    if path.parts[:3] == ('deployment', '.work', 'rootfs'):
        return None
    return path


class Collector:
    def __init__(self, src, out):
        self.src = src
        self.out = out
        self.files = defaultdict(set)
        self.refs = defaultdict(lambda: defaultdict(set))
        self.snapshots = {}
        self.pending = deque()
        self.processed = set()
        self.external = set()
        self.aliases = {}
        self.missing_optional = []

    def add(self, path, reason, expected=None):
        if isinstance(path, Path):
            path = path.as_posix()
        q = relpath(path, self.src)
        if q is None:
            if expected and isinstance(path, str) and path.startswith('/'):
                self.external.add((path, expected, reason))
            return
        if expected:
            self.refs[q][expected].add(reason)
        self.files[q].add(reason)
        if q.suffix == '.json' and q not in self.processed:
            self.pending.append(q)

    def directory(self, folder, reason, *, code=False):
        for file in sorted((self.src / folder).rglob('*')):
            if file.is_file() and (not code or file.suffix in SOURCE or file.name in {'Dockerfile', 'LICENSE', 'COPYING'}):
                self.add(file.relative_to(self.src), reason)

    def read(self, rel):
        f = self.src / rel
        if f.is_file() and f.stat().st_size <= 16 * 1024**2:
            return json.loads(f.read_bytes())
        return {}

    def hash_references(self, data, reason):
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(x for x in node if isinstance(x, (dict, list)))
                continue
            if not isinstance(node, dict):
                continue
            for key, value in node.items():
                if key in KEYS and isinstance(value, dict):
                    for name, digest in value.items():
                        if isinstance(digest, str) and HASH.fullmatch(digest):
                            if '/' in name or (self.src / name).is_file():
                                self.add(name, reason, digest)
                    continue
                if isinstance(value, (dict, list)):
                    stack.append(value)
            digest = node.get('sha256')
            if isinstance(digest, str) and HASH.fullmatch(digest):
                for label in ('path', 'file', 'protocol', 'source', 'receipt', 'proof'):
                    name = node.get(label)
                    if isinstance(name, str) and (name.startswith('/') or (self.src / name).is_file()):
                        self.add(name, reason, digest)
                        break

    def follow(self):
        # Follow bindings in the explicitly selected records. Their full
        # direct binding maps remain intact in the snapshot. Historical
        # training lineage can point at entire unrelated experiments.
        for _ in range(len(self.pending)):
            rel = self.pending.popleft()
            if rel in self.processed:
                continue
            self.processed.add(rel)
            if rel.parts[:2] == ('deployment', 'backups') or rel.name in {'artifact_manifest.json', 'source_manifest.json'}:
                continue
            try:
                self.hash_references(self.read(rel), str(rel))
            except (UnicodeError, json.JSONDecodeError):
                pass

    def select(self):
        self.directory(GATE, '冻结门控原始快照')
        self.directory(TTA, 'TTA093 冻结清单与校验器')
        gate = self.read(GATE / 'manifest.json')
        for row in gate['files']:
            original = Path(row['project_relative_path'])
            snapshot = GATE / row['snapshot']
            self.snapshots[(original, row['sha256'])] = self.src / snapshot
            self.add(original, '全事件门控 333 项归档', row['sha256'])
        for row in gate['reference_registries']:
            snapshot = GATE / row['snapshot']
            original = relpath(row['source'], self.src)
            if original is not None:
                self.snapshots[(original, row['sha256'])] = self.src / snapshot
                self.add(original, '全事件门控原始参考计划', row['sha256'])
        for path in (
            'README.md', '.gitattributes',
            'deployment/Dockerfile', 'deployment/run.sh', 'deployment/run.py', 'deployment/bootstrap.sh',
            'deployment/environment.json', 'deployment/task_adapter_python.sh',
            'multi_event_gate_v1/scripts/eval_event_full_worker.sh', 'multi_event_gate_v1/task_adapter_v1/scripts/eval.sh',
            'multi_event_gate_v1/cx_tta/evaluate_frozen_event.py', 'multi_event_gate_v1/cx_tta/frozen_event_policy.py',
            'multi_event_gate_v1/cx_tta/multi_event_policy.py', 'multi_event_gate_v1/cx_tta/stacking_policy.py',
            'deployment/doctor.py', 'deployment/install_overlay.py', 'deployment/artifact_manifest.json',
            'deployment/source_manifest.json', 'deployment/CONTINUE_ON_OTHER_SERVER.md',
            'deployment/user_container.py', 'deployment/gpu_doctor.py',
            'deployment/adapter_cache_cpu.py', 'deployment/startup_loading.py',
            'multi_event_gate_v1/docs/全事件候选门控方案冻结说明_v1_20261101.md',
        ):
            if (self.src / path).is_file():
                self.add(path, '部署和冻结说明')
        for glob in ('candidate_eval*.py', 'target*.sh', 'requirements*', 'constraints*', 'Dockerfile*'):
            for p in (self.src / D).glob(glob):
                if p.is_file():
                    self.add(p.relative_to(self.src), '原部署和 event_full 运行入口')
        for folder in ('configs', 'deployment/payload/licenses'):
            self.directory(Path(folder), '部署配置与许可')
        for file in (self.src / D / 'reports').glob('target_cross_task_shared250_a050_*20261025.*'):
            self.add(file.relative_to(self.src), '四任务 event_full 400 回合验收')
        for name in ('TTA093_FROZEN_20261101.md', 'TTA093_FOUR_TASK100_20261101.md',
                     'TTA093_THREE_TASK_PROVENANCE_20261102.md', 'SEVEN_TASK_TTA_GATE_PLAN_20261110.md'):
            self.add(T / 'docs' / name, '冻结方案和结果说明')
        for root in ('native_seven_official_v9', 'native_seven_campaign_v8', 'native_seven_gated_continue_v1',
                     'native_seven_window_watch_v4', 'native_seven_report_v7', 'native_seven_resource_readout_v6',
                     'native_fit7_postexit_recovery_v2', 'gated_postexit_recovery_v2'):
            self.directory(T / root, '七任务当前执行、资格、守卫或结果汇总代码')
        for root in ('native_seven_official_v9/gated_000', 'native_seven_official_v9/plain_001',
                     'native_calibrated_five100_v33/official_019_p1', 'native_cross_task100_v19'):
            for name in ('study.json', 'pilot_complete.json', 'actual_host_wait_exit.json', 'execution_state.json'):
                rel = T / 'outputs' / root / name
                if (self.src / rel).is_file():
                    self.add(rel, 'TTA 已冻结运行和退出记录')
        for suffix in (
            'native_fit7_postexit_recovery_v2/qualification.json',
            'native_fit7_postexit_recovery_v2/response_step1000.pt',
            'native_fit7_postexit_recovery_v2/experiment_summary.json',
            'native_calibrated_five100_v33/official100_complete.json',
            'gated_postexit_recovery_v2/qualified_TRAIN/qualification.json',
            'native_seven_official_v9/supervision_plain/qualified_012.json',
            'native_seven_gated_continue_v1/failure.json',
            'native_seven_official_v9/gated_000/campaign_failure.json',
        ):
            self.add(T / 'outputs' / suffix, 'TTA 权重、资格与已验收结果')
        for row in ('gated_000', 'plain_001'):
            rel = T / 'outputs/native_seven_official_v9' / row / 'study.json'
            d = self.read(rel)
            for k in ('trained_qualification', 'projection'):
                self.add(d[k], str(rel))
            train = relpath(d['training'], self.src)
            for name in ('adapter_seed0.pt','training_plan.json','training_result.json','training_audit.json','cache_manifest.json'):
                self.add(train / name, 'StackingAdapter 初始化及其运行时完整性检查')
            tp = self.read(train / 'training_plan.json')
            for k in ('projection','robot_config'):
                if k in tp:
                    self.add(tp[k], str(train / 'training_plan.json'))
            runtime = relpath(d['runtime_source'], self.src)
            for name in ('envs','task_config','description'):
                self.directory(runtime / name, '原生 RoboTwin 仿真源码与任务配置')
            self.external.add((str(runtime / 'assets'), None, 'RoboTwin 原生仿真 assets 的外部挂载'))
        self.follow()

    def static_imports(self):
        """Resolve repository local imports missed by the recorded execution manifests."""
        direct = (Path('.'), W, T, D, D / '.work', T / 'resource_guard_v1')
        extra = set()
        for q in self.files:
            if q.suffix == '.py' and q.is_relative_to(T) and len(q.parts) > len(T.parts) + 2:
                up = Path(*q.parts[:len(T.parts) + 1])
                if any(part.startswith('cx_tta') for part in q.parts[len(T.parts)+1:-1]):
                    extra.add(up)
        roots = direct + tuple(sorted(extra))
        @lru_cache(maxsize=350_000)
        def exists(path, is_dir=False):
            p=self.src/path
            return p.is_dir() if is_dir else p.is_file()
        starts = [q for q in self.files if q.suffix == '.py' and (
            q.parent == D or q.is_relative_to(T/'native_seven_official_v9') or
            q.is_relative_to(T/'native_cross_task100_v19') or str(q) in {
                'multi_event_gate_v1/cx_tta/evaluate_frozen_event.py',
                'multi_event_gate_v1/cx_tta/frozen_event_policy.py',
                'multi_event_gate_v1/cx_tta/multi_event_policy.py',
                'multi_event_gate_v1/cx_tta/stacking_policy.py',
            })]
        queue = deque(sorted(starts))
        scanned = set()
        while queue:
            rel = queue.popleft()
            p = self.src / rel
            if rel in scanned or rel.suffix != '.py' or not p.is_file() or rel.is_relative_to(GATE):
                continue
            scanned.add(rel)
            try:
                tree = ast.parse(p.read_bytes())
            except SyntaxError:
                continue
            def add_if_exists(path):
                if exists(path) and relpath(str(path), self.src):
                    if path not in scanned:
                        queue.append(path)
                    self.add(path, '静态导入/动态导入字面量补足')
                    return True
                return False
            def resolve(prefix, names, root):
                for i in range(1, len(prefix)+1):
                    add_if_exists(root.joinpath(*prefix[:i], '__init__.py'))
                base = root.joinpath(*prefix)
                found = add_if_exists(base.with_suffix('.py')) if prefix else False
                if exists(base, True):
                    found = add_if_exists(base / '__init__.py') or found
                    for name in names:
                        if name != '*':
                            found = add_if_exists(base / (name+'.py')) or found
                            found = add_if_exists(base / name / '__init__.py') or found
                return found
            active_roots=(rel.parent,*direct,*sorted(extra)) if rel.is_relative_to(T) else (rel.parent,*direct)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    prefix = node.module.split('.') if node.module else []
                    names = [v.name for v in node.names]
                    if node.level:
                        base = rel.parent
                        for _ in range(node.level-1):
                            base = base.parent
                        resolve(prefix,names,base)
                    else:
                        for root in active_roots:
                            resolve(prefix,names,root)
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        for root in active_roots:
                            resolve(name.name.split('.'),[],root)
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    name = node.args[0].value
                    fn = node.func
                    target = fn.attr if isinstance(fn,ast.Attribute) else (fn.id if isinstance(fn,ast.Name) else '')
                    if target in {'import_module','__import__'} and isinstance(name,str) and re.fullmatch(r'[a-zA-Z_][\w.]*',name):
                        for root in active_roots:
                            resolve(name.split('.'),[],root)

    def inventory(self):
        entries = {}
        issues = []
        for q in sorted(self.files):
            file = self.src / q
            if not file.is_file():
                candidates = [p for (rel, _), p in self.snapshots.items() if rel == q and p.is_file()]
                if not candidates:
                    issues.append({'path': str(q), 'issue': 'missing', 'expected': sorted(self.refs[q]), 'reasons': sorted(self.files[q])})
                    continue
                file = candidates[0]
            actual = sha(file)
            expected = self.refs[q]
            if expected and actual not in expected:
                f = next((self.snapshots[(q, h)] for h in expected if (q,h) in self.snapshots and self.snapshots[(q,h)].is_file() and sha(self.snapshots[(q,h)]) == h),None)
                if f is not None:
                    file,actual=f,sha(f)
            for h,reasons in sorted(expected.items()):
                if h == actual:
                    continue
                alt = self.snapshots.get((q,h))
                if alt is not None and alt.is_file() and sha(alt)==h:
                    self.aliases[str(q) + '@' + h] = str(alt.relative_to(self.src))
                else:
                    issues.append({'path': str(q), 'issue': 'historical digest unavailable', 'expected': h, 'preserved': actual, 'reasons': sorted(reasons)})
            issues_in_file = any(part in ('.ssh','.codex','.git','.aws') for part in file.resolve().parts)
            if (q.suffix in TEXT or not q.suffix) and not issues_in_file:
                if file.stat().st_size < 256 * 1024**2:
                    issues_in_file = SECRET.search(file.read_bytes()) is not None
                else:
                    with file.open('rb') as stream:
                        for line in stream:
                            if SECRET.search(line):
                                issues_in_file = True
                                break
            if issues_in_file:
                issues.append({'path': str(q), 'issue': 'credential-pattern or private path excluded', 'expected': actual})
                continue
            entries[q] = (file, actual, file.stat().st_size)
        return entries, issues

    def run(self, inventory_only):
        self.select()
        print(f'Freeze and evidence paths collected: {len(self.files)}', flush=True)
        self.static_imports()
        print(f'After local import closure: {len(self.files)}', flush=True)
        entries, issues = self.inventory()
        inventory = {'source_project': str(self.src), 'original_project_aliases': list(PROJECT_PREFIXES),
                     'source_head': subprocess.check_output(['git','-C',str(self.src),'rev-parse','HEAD'],text=True).strip(),
                     'files': [{'path': str(q), 'sha256': h, 'bytes': n, 'reasons': sorted(self.files[q]),
                                'layer': 'evidence' if self.evidence(q,n) else 'git'} for q, (_,h,n) in entries.items()],
                     'historical_versions_in_frozen_snapshot': self.aliases,'unresolved_bindings': issues,
                     'external': [{'path': p,'sha256': h,'referenced_by': reason} for p,h,reason in sorted(self.external,key=str)]}
        summary = {'files': len(entries), 'bytes': sum(x[2]for x in entries.values()),
                   'by_layer': dict(Counter(row['layer']for row in inventory['files'])),
                   'issues': dict(Counter(row['issue']for row in issues)),
                   'external': len(inventory['external'])}
        inventory['summary'] = summary
        name = 'inventory-preview.json' if inventory_only else 'inventory.json'
        (self.out / name).write_text(json.dumps(inventory,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(summary),flush=True)
        if inventory_only:
            return
        assert not any(x['issue']=='credential-pattern or private path excluded' for x in issues), 'Review exclusions before assembling'
        for q,(source,h,n) in entries.items():
            top = self.out / ('_evidence_staging' if self.evidence(q,n) else 'project')
            target = top / q
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            if sha(target) != h:
                raise RuntimeError(f'Source changed while copying {q}')

    @staticmethod
    def evidence(path, size):
        return size > 2 * 1024**2 or path.suffix in {'.hdf5','.jsonl','.log','.bin','.so','.npz'} and size > 128*1024


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path)
    parser.add_argument('--inventory-only',action='store_true')
    args=parser.parse_args()
    Collector(args.project.resolve(),Path(__file__).resolve().parents[1]).run(args.inventory_only)
