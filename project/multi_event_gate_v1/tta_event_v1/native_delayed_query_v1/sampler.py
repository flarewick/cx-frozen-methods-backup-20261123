"""One additional natural query, from the same live owner RNG after following32.

The old protocol remains unchanged. No archived RNG or new handoff is imported.
"""
from native_nextchunk_v1.sampler_current import CurrentQueryFork, CurrentBranch
from native_nextchunk_v1.sampler_fork import audited_draw

class DelayedQueryFork(CurrentQueryFork):
    def branch(self, name, router, lane, *, device_uuid, model_sha256):
        self._identity(router, device_uuid=device_uuid, model_sha256=model_sha256)
        if name in self._branches or not name or not self.ledger.awaiting_execution or self._after is None:
            raise ValueError('Unique fresh selected-query branch required')
        self._branches.add(name)
        return DelayedBranch(self, name, lane)

class DelayedBranch(CurrentBranch):
    def __init__(self, source, name, lane):
        super().__init__(source, name, lane)
        self.delayed_draws = 0

    def permit_delayed(self, *, query, tick, device_uuid, model_sha256):
        self._identity(device_uuid, model_sha256)
        if not self.installed or self.following_draws != 1 or self.delayed_draws:
            raise ValueError('One delayed natural query after the first following32 only')
        self.ledger.require_following_query(tick)
        self.ledger.permit_query(query, tick)

    def draw_delayed(self, operation, *, query, tick, device_uuid, model_sha256):
        self.permit_delayed(query=query, tick=tick, device_uuid=device_uuid, model_sha256=model_sha256)
        try:
            result, _, note = audited_draw(
                self.source.router, self.lane, self.source.model, self.ledger,
                operation, query=query, tick=tick)
        except BaseException:
            self.poisoned = True
            raise
        self.delayed_draws += 1
        return result, note

    def finish_delayed(self, returned, *, end_tick):
        if self.poisoned or self.delayed_draws != 1:
            raise ValueError('No valid delayed query to finish')
        self.ledger.finish_chunk(returned, end_tick=end_tick)
