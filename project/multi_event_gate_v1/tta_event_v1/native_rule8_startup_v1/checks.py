"""Unmodified analytic v15 policy fixtures, excluding LIBERO-specific results."""
from copy import deepcopy
from datetime import datetime
import os
from zoneinfo import ZoneInfo
from .policy import daytime,thresholds,require_admission,evaluate
from native_governor180_v4.statistics import means,parse_pmon
checks=[]
def check(name, condition):
    assert condition,name
    checks.append(name)

def wall(hour):
    return datetime(2026,11,9,hour,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()

def stats(util=70, memory_pct=50):
    return dict(util=util,sm=util,bandwidth=0,memory_pct=memory_pct,memory_mib=4096.,
                full_window=True,metric_gap_seconds=0,cpu=100,rss_max=1024)

def fleet(free=2, util=70, memory=50, hour=10):
    cards=[dict(index=i,uuid=str(i),own_pids=[i+1],pids=[i+1],available=False,
                own_rolling180=stats(util,memory),processes=[])for i in range(2)]
    cards += [dict(index=i+2,uuid=str(i+2),own_pids=[],pids=[],available=True,
                  own_rolling180=stats(0,0),processes=[])for i in range(free)]
    return dict(cards=cards,available_indices=list(range(2,2+free)),own_gpu_count=2,wall_time=wall(hour))

def one_job(f, age=300, util=35, memory=4096, cpu=40):
    p=dict(pid=10,birth=100,uid=os.getuid(),age=age,rss=1024,
           rolling180=dict(stats(util),memory_mib=memory,cpu=cpu))
    f['cards'][0]['processes']=[p]
    return p

def bad(f):
    return evaluate(f,{(10,100)})['reasons']

for n,u,m,eu,em in [(0,35,4096,70,50),(2,35,4096,70,50),(3,30,3072,60,30),
                     (4,30,3072,60,30),(5,25,2048,50,20),(7,25,2048,50,20)]:
    t=thresholds(n);check('free_tier_'+str(n),(t['util'],t['memory_mib'],t['extra_util'],t['extra_memory_pct'])==(u,m,eu,em))
check('07_night',not daytime(wall(7)));check('08_day',daytime(wall(8)))
check('22_day',daytime(wall(22)));check('23_night',not daytime(wall(23)))
rows=[dict(monotonic=0.,sm=100,bandwidth=0,memory_mib=4096,metrics_available=True),
      dict(monotonic=3.,sm=0,bandwidth=100,memory_mib=4096,metrics_available=True)]
check('MAX_of_component_means_not_mean_of_MAX',means(rows,3)['util']==50)
rows[1].update(sm=None,bandwidth=None,metrics_available=False)
check('missing_components_never_substituted',means(rows,3)['util']==50)
check('missing_clock_measured',means(rows,12)['metric_gap_seconds']==12)
check('pmon_components_preserved',parse_pmon('2 10 C 40 80 - - python')[(2,10)]==[40,80])
f=fleet();one_job(f);check('exact35_CPU40_boundaries_pass',not bad(f))
for field,value,rule in [('util',34.999,'Rule4'),('memory_mib',4095.999,'Rule4'),('cpu',39.999,'Rule5'),('cpu',1600.001,'Rule5')]:
    g=deepcopy(f);g['cards'][0]['processes'][0]['rolling180'][field]=value
    check(field+str(value)+'_rejected',any(rule in x for x in bad(g)))
g=deepcopy(f);g['cards'][0]['processes'][0]['rolling180'].update(util=0,memory_mib=0,cpu=0)
g['cards'][0]['processes'][0]['age']=279.999
check('precheck_not_before280',not bad(g))
g['cards'][0]['processes'][0]['age']=280
check('precheck_stops_before300',any('Rule4'in x for x in bad(g)))
g=deepcopy(f);g['cards'][0]['processes'][0]['age']=36*3600
check('36_hour_limit',any('Rule1'in x for x in bad(g)))
g=deepcopy(f);g['cards'][0]['processes'][0]['rolling180']['rss_max']=128*2**30+1
check('128GiB_peak_RSS',any('Rule3'in x for x in bad(g)))
g=deepcopy(f);g['cards'][0]['processes'][0]['rolling180']['metric_gap_seconds']=9.01
check('missing_metric_gap_rejected',any('9s'in x for x in bad(g)))
g=deepcopy(f);g['cards'][1]['processes']=deepcopy(g['cards'][0]['processes'])
check('one_PID_two_GPUs_rejected',any('Rule2'in x for x in bad(g)))
g=fleet();one_job(g);g['cards'][0]['own_rolling180']=stats(48.2,35.1);g['cards'][1]['own_rolling180']=stats(59.3,79.7)
check('actual_governor_4171573_rule8_regression',any('Rule8'in x for x in bad(g)))
g['cards'][1]['own_rolling180']=stats(70,50)
check('one_low_GPU_exception_allowed',not bad(g))
g['cards'][1]['own_rolling180']=stats(99,49.99)
check('high_util_does_not_replace_VRAM_AND',any('Rule8'in x for x in bad(g)))
g=fleet(free=3,util=65,memory=40);one_job(g,util=30,memory=3072)
check('3_free_looser_tier_pass',not bad(g))
g['available_indices']=[2,3]
check('dynamic_3_to2_free_rechecked',any('Rule8'in x for x in bad(g)))
g=fleet(free=3,util=65,memory=40)
try:require_admission(g,'2')
except RuntimeError:check('postlaunch_free_tier_not_prelaunch',True)
else:raise AssertionError('Incorrectly admitted against prelaunch tier')
g=fleet(free=4,util=60,memory=30);check('admit_exact_postlaunch60_AND30',require_admission(g,'2')['index']==2)
g['cards'][0]['own_rolling180']['full_window']=False
try:require_admission(g,'2')
except RuntimeError:check('instantaneous_only_admission_rejected',True)
else:raise AssertionError('Missing180s history admitted')
g=fleet(free=2,util=0,memory=0,hour=23);one_job(g)
check('night_rule8_disabled',not bad(g))
check('night_extra_card_allowed_within_user_cap',require_admission(g,'2')['index']==2)
g['cards'][2].update(own_pids=[9],available=False)
try:require_admission(g,'3')
except RuntimeError:check('night_user_max3_retained',True)
else:raise AssertionError('User max3 violated')
g=fleet();one_job(g);foreign=dict(pid=11,birth=99,uid=os.getuid()+1,age=1000,rss=1024,rolling180=stats(50))
g['cards'][0]['processes'].append(foreign);check('earlier_foreign_exact50_allowed',not bad(g))
foreign['rolling180']['util']=50.01;check('earlier_foreign_above50_stop',any('Rule6'in x for x in bad(g)))
foreign['birth']=101;check('later_foreign_not_prior',not bad(g))

if __name__=='__main__':print('CURRENT_POLICY_ANALYTIC_CHECKS_PASS',len(checks),flush=True)
