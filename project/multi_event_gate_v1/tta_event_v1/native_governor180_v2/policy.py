"""2026-11-09 governor policy. No CUDA, model or simulation dependencies."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

L = Path(__file__).resolve().parents[1].parent / 'libero'
RULES = Path('/home/qiufw/服务器使用注意事项.md')
RULES_SHA = '82eda4d4ca4c747677f273fef53d28ac655dafcfc940569ae64f58b419b60e7d'
WINDOW = 180
MAX_GPUS = 3  # User's additional all-hours cap.


def verify_rules():
    assert hashlib.sha256(RULES.read_bytes()).hexdigest() == RULES_SHA, 'Server rules changed'
    Path('/home/qiufw/GPU使用与优化复用指南.md').read_text()


def daytime(unix):
    return 8 <= datetime.fromtimestamp(unix, ZoneInfo('Asia/Shanghai')).hour < 23


def thresholds(free):
    assert isinstance(free, int) and free >= 0
    if free <= 2:
        return dict(util=35., memory_mib=4096., extra_util=70., extra_memory_pct=50.)
    if free <= 4:
        return dict(util=30., memory_mib=3072., extra_util=60., extra_memory_pct=30.)
    return dict(util=25., memory_mib=2048., extra_util=50., extra_memory_pct=20.)


def extra_ok(card, limits):
    s = card.get('own_rolling180', {})
    return bool(s.get('full_window') and s.get('metric_gap_seconds', 999) <= 9
                and s['util'] >= limits['extra_util']
                and s['memory_pct'] >= limits['extra_memory_pct'])


def require_admission(fleet, uuid):
    card = next((c for c in fleet['cards'] if c['uuid'] == uuid), None)
    if not card or not card['available']:
        raise RuntimeError('Selected GPU is not idle')
    own = [c for c in fleet['cards'] if c['own_pids']]
    if len(own) >= MAX_GPUS:
        raise RuntimeError('All-hours user cap of3 GPUs')
    # Starting this job consumes an idle GPU. Do not admit against the easier
    # pre-launch tier or assume the new job will become a high-load exception.
    limits = thresholds(max(0, len(fleet['available_indices']) - 1))
    if daytime(fleet['wall_time']) and own:
        if not all(extra_ok(c, limits) for c in own):
            raise RuntimeError('Rule8: existing GPUs need180s paired utilization AND VRAM thresholds')
    return card


def evaluate(fleet, owned, *, precheck=True):
    """Return violations concerning THIS job, never terminate another job."""
    reasons = []
    own_cards = [c for c in fleet['cards'] if c['own_pids']]
    limits = thresholds(len(fleet['available_indices']))
    if len(own_cards) > MAX_GPUS:
        reasons.append('User all-hours3 GPU cap')
    job = [(c, p) for c in fleet['cards'] for p in c['processes']
           if (p['pid'], p['birth']) in owned]
    per_pid_cards = {}
    for c, p in job:
        per_pid_cards.setdefault((p['pid'], p['birth']), set()).add(c['uuid'])
        if len(per_pid_cards[(p['pid'], p['birth'])]) > 1:
            reasons.append('Rule2: one PID uses multiple GPUs')
        if p['age'] >= 36 * 3600:
            reasons.append('Rule1:36 hour process lifetime')
        s = p['rolling180']
        # Immediate project RSS ceiling is stricter than the governor grace.
        if p['rss'] > 128 * 2**30:
            reasons.append('Project immediate128GiB RSS ceiling')
        for other in c['processes']:
            if other['uid'] != p['uid'] and other['birth'] < p['birth']:
                if not other['rolling180'].get('full_window') or other['rolling180']['util'] > 50:
                    reasons.append('Rule6: earlier foreign process above50% or unverified')
        check_age = 280 if precheck else 300
        if p['age'] >= check_age:
            if s.get('metric_gap_seconds', 999) > 9:
                reasons.append('Project9s process-metric gap')
            if not s.get('full_window'):
                reasons.append('Unverified180s process window')
            if s['util'] < limits['util'] or s['memory_mib'] < limits['memory_mib']:
                reasons.append('Rule4: rolling180s GPU utilization/VRAM')
            if not 40 <= s['cpu'] <= 1600:
                reasons.append('Rule5: rolling180s CPU')
            if s['rss_max'] > 128 * 2**30:
                reasons.append('Rule3:180s maximum RSS')
            if daytime(fleet['wall_time']) and len(own_cards) > 1:
                below = [c['index'] for c in own_cards if not extra_ok(c, limits)]
                if len(below) > 1:
                    reasons.append('Rule8: more than one user GPU fails paired180s thresholds')
    return dict(passed=not reasons, reasons=sorted(set(reasons)), limits=limits,
                daytime=daytime(fleet['wall_time']), job_pids=[p['pid'] for _, p in job])
