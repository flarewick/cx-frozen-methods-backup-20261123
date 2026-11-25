"""Archive compiler cache after an actual q0 passes unchanged original-reference checks."""
from .actual_query_compiler_cache_v151 import publish
from .matched_actor import AuditedActor as Original


class AuditedActor(Original):
    def query(self, observed, record, folder):
        result = super().query(observed, record, folder)
        if record['query'] == 0:
            root = next(p for p in folder.parents if (p/'official_compiled_sampler_restore.json').is_file())
            self.c.call(self.sampler,publish,root,folder/'query000_original_G_match.json')
        return result
