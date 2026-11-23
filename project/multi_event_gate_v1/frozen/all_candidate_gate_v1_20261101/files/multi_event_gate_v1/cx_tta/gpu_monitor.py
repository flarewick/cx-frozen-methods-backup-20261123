"""Telemetry outages pause admission; they do not terminate healthy workers."""
import math
import subprocess
import time


class GpuMonitor:
    def __init__(self, query, *, initial_retries=3, retry_delay=2.):
        self.query=query;self.initial_retries=initial_retries;self.retry_delay=retry_delay
        self.last=None;self.last_time=None;self.failures=0;self.total_failures=0

    def observe(self):
        try:
            value=self.query()
            for key in ('total_mib','used_mib','free_mib','utilization'):
                if not math.isfinite(value[key]) or value[key]<0:
                    raise ValueError('Invalid GPU telemetry '+key)
            if not value.get('uuid') or value['total_mib']<=0:
                raise ValueError('GPU identity/capacity unavailable')
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,ValueError) as exc:
            self.failures+=1;self.total_failures+=1
            if self.last is None:return None
            return dict(self.last,free_mib=0,process_mib={},telemetry_fresh=False,
                telemetry_error=repr(exc),consecutive_failures=self.failures,
                last_good_age_s=time.monotonic()-self.last_time)
        if self.last and value['uuid']!=self.last['uuid']:
            raise RuntimeError('Selected GPU identity changed; refusing a different card')
        self.last=dict(value);self.last_time=time.monotonic();self.failures=0
        return dict(value,telemetry_fresh=True,last_good_age_s=0,consecutive_failures=0)

    def initial(self):
        for n in range(self.initial_retries):
            value=self.observe()
            if value is not None and value['telemetry_fresh']:return value
            if n+1<self.initial_retries:time.sleep(self.retry_delay)
        raise RuntimeError('GPU telemetry unavailable before launch; no worker started')
