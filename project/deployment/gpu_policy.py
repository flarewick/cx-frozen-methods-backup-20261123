"""Account-wide, all-hours GPU admission; no warmup or night-time exceptions."""
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

MAX_GPUS = 3


def parse_inventory(xml, owner):
    cards=[]
    for index,gpu in enumerate(ET.fromstring(xml).findall('gpu')):
        number=lambda field:int(gpu.findtext(field).split()[0])
        pids=[int(p.findtext('pid')) for p in gpu.findall('./processes/process_info')]
        ours=[pid for pid in pids if owner(pid)]
        memory=number('./fb_memory_usage/used');sm=number('./utilization/gpu_util')
        cards.append(dict(index=index,uuid=gpu.findtext('uuid'),memory_mib=memory,
            total_mib=number('./fb_memory_usage/total'),sm=sm,pids=pids,own_pids=ours,
            available=not pids and memory<=64 and sm<=5))
    if not cards:raise RuntimeError('No GPUs in resource inventory')
    return dict(max_gpus=MAX_GPUS,cards=cards,own_gpu_count=sum(bool(c['own_pids']) for c in cards),
        available_indices=[c['index'] for c in cards if c['available']])


def inventory():
    def owner(pid):
        try:return Path('/proc',str(pid)).stat().st_uid==os.getuid()
        except FileNotFoundError:return False
    result=subprocess.run(['nvidia-smi','-q','-x'],check=True,stdout=subprocess.PIPE,text=True,timeout=3)
    return parse_inventory(result.stdout,owner)


def require_admission(fleet,uuid):
    selected=next((c for c in fleet['cards'] if c['uuid']==uuid),None)
    if selected is None or not selected['available']:
        raise RuntimeError('Selected GPU is no longer available; preserve existing jobs')
    if fleet['own_gpu_count']>=MAX_GPUS:
        raise RuntimeError('All-hours account GPU limit is 3; refusing another GPU')
    for card in fleet['cards']:
        if card['own_pids'] and card['sm']<75 and card['memory_mib']*2<card['total_mib']:
            raise RuntimeError('Existing account GPU does not meet extra-GPU admission threshold')
    return selected


def parse_process_utilization(text):
    rows={}
    for line in text.splitlines():
        fields=line.split()
        if len(fields)<5 or not fields[0].isdigit() or not fields[1].isdigit():continue
        if not fields[3].isdigit() or not fields[4].isdigit():continue
        sm,bandwidth=int(fields[3]),int(fields[4])
        if not 0<=sm<=100 or not 0<=bandwidth<=100:raise ValueError('Invalid process utilization')
        rows[int(fields[1])]=max(sm,bandwidth)
    return rows


def process_utilization(index):
    result=subprocess.run(['nvidia-smi','pmon','-i',str(index),'-s','u','-c','1'],
        check=True,stdout=subprocess.PIPE,text=True,timeout=3)
    return parse_process_utilization(result.stdout)
