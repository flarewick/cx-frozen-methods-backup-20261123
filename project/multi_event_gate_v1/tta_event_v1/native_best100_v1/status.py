from .common import O,read
import json
value={name:read(O/name)for name in ('progress.json','live_status.json','complete_comparison.json','campaign_failure.json')if(O/name).exists()}
print(json.dumps(value,ensure_ascii=False,indent=2))
