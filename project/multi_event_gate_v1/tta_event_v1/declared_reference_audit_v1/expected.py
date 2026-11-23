"""Exactly the declared first-case controls plus required regenerated references."""
def expected(cold_rows,accepted,regenerated,*,mandatory_first_case):
    if type(mandatory_first_case)is not bool:raise ValueError('Explicit reference contract required')
    rows=accepted if mandatory_first_case else [x for x in accepted if x['case_id']in{r['case_id']for r in cold_rows}]
    first={}
    for x in rows:
        if x['task']not in first or x['episode_index']<first[x['task']]['episode_index']:first[x['task']]=x
    return {x['case_id']for x in first.values()}|set(regenerated)

def validate(actual,cold_rows,accepted,regenerated,*,mandatory_first_case):
    wanted=expected(cold_rows,accepted,regenerated,mandatory_first_case=mandatory_first_case)
    assert set(actual)==wanted and len(actual)==len(wanted),('Missing, duplicate or unexpected reference',actual,wanted)
    return wanted
