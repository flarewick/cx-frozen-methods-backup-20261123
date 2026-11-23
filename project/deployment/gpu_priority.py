"""Prove GPU arrival from observations, never from unrelated process birth times."""
class ArrivalOrder:
    def __init__(self):
        self.own_first_seen=None
        self.foreign_first_seen={}
    def observe(self,seconds,active,foreign):
        active=set(active);foreign=set(foreign)
        before=self.own_first_seen
        for pid in foreign:self.foreign_first_seen.setdefault(pid,seconds)
        if active and not foreign and before is None:self.own_first_seen=seconds
        # An earlier exclusive GPU sample proves our order. Simultaneous first
        # sighting cannot establish priority; CPU-process start time is irrelevant.
        verified=bool(active and before is not None)
        return dict(own_first_seen_seconds=self.own_first_seen,
                    foreign_first_seen_seconds=dict(self.foreign_first_seen),
                    foreign_pids=sorted(foreign),own_arrival_precedes_foreign=verified if foreign else None,
                    stop_for_prior_or_unverified_occupant=bool(foreign and not verified))
