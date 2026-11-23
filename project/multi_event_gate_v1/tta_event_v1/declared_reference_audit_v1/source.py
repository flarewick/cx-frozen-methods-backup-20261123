"""Retain every original closure/provenance check; bind the declared reference set."""
import inspect
from pathlib import Path
from .expected import validate

def make(original):
    before=inspect.getsource(original)
    old=""" cold_controls=[p for p in controls if p.parent.name not in regenerated]
 assert len(cold_controls)==len({r['case_id'].split(':')[0]for r in rows})
 assert {p.parent.name for p in controls}=={p.parent.name for p in cold_controls}|regenerated
 assert len(controls)==len(cold_controls)+len(regenerated)"""
    new=""" accepted=[read(p)for p in(root/'accepted').glob('*.json')]
 _validate_references([p.parent.name for p in controls],rows,accepted,regenerated,mandatory_first_case=spec.get('mandatory_first_case_whole_reference',False))
 cold_controls=[p for p in controls if p.parent.name not in regenerated]"""
    assert before.count(old)==1
    changed=before.replace(old,new);assert changed.replace(new,old)==before
    namespace=dict(original.__globals__,_validate_references=validate)
    exec(compile(changed,__file__,'exec'),namespace)
    fn=namespace[original.__name__];fn._source_before=before;fn._source_after=changed;return fn
