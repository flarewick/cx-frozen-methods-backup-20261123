"""Pure control-flow contracts; not integrated/verified with a CUDA sampler yet."""
from dataclasses import dataclass,field

@dataclass
class QueryLedger:
    branch: str
    records: list=field(default_factory=list)
    awaiting_execution: bool=False
    origin_seen: bool=False
    origin_ready: bool=False
    stage_tick: int|None=None
    censored: bool=False
    def permit_query(self,query,tick):
        if self.censored or self.awaiting_execution:
            raise ValueError('Cannot draw after censor or before prior chunk execution ends')
        if type(query)is not int or query!=len(self.records) or type(tick)is not int or tick<0:
            raise ValueError('Contiguous actual query and nonnegative clock required')
        if self.records and tick<self.records[-1]['tick']:
            raise ValueError('Actual clock went backwards')
        if self.records and tick<self.records[-1]['end_tick']:
            raise ValueError('New observation cannot precede completed execution')
        if self.origin_seen and not self.origin_ready:
            raise ValueError('Original stage95 receipt unknown, no following query')
        return True
    def record_draw(self,query,tick,base_sha256,rng_before,rng_after,*,actual_sampler_call):
        self.permit_query(query,tick)
        if actual_sampler_call is not True or rng_before==rng_after:
            raise ValueError('Cached actions never substitute for one real internal-noise draw')
        if any(not isinstance(v,str) or len(v)!=64 for v in (base_sha256,rng_before,rng_after)):
            raise ValueError('Full explicit SHA256 identities required')
        self.records.append(dict(query=query,tick=tick,base_sha256=base_sha256,
                                 rng_before=rng_before,rng_after=rng_after))
        self.awaiting_execution=True
    def finish_chunk(self,returned,*,end_tick,requested95_arrived=False,origin=False):
        if not self.awaiting_execution or type(returned)is not int or not 0<=returned<=32:
            raise ValueError('Exactly one prior draw and truthful returned count required')
        if type(end_tick)is not int or end_tick<self.records[-1]['tick']:
            raise ValueError('Actual execution-end clock required')
        if origin and self.origin_seen:raise ValueError('One original origin only')
        self.records[-1].update(returned=returned,end_tick=end_tick)
        self.awaiting_execution=False
        if returned<32:self.censored=True
        if origin:
            self.origin_seen=True
            self.origin_ready=returned==32 and requested95_arrived is True
            self.stage_tick=end_tick if self.origin_ready else None
    def require_following_query(self,tick):
        if self.censored or self.awaiting_execution or not self.origin_ready:
            raise ValueError('No later query before full original stage receipt')
        if type(tick)is not int or tick<self.stage_tick:
            raise ValueError('Following RGB must actually arrive after stage')
        return len(self.records)
