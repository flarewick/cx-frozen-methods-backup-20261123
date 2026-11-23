"""One causal decision packet. No scene IDs, outcomes, GT, or future fields."""
import numpy as np
from .local_benefit_cache import legal_packet
from .structured_action_adapter import FORWARD_KEYS, ARM_INDICES

INPUT_KEYS = FORWARD_KEYS | {'base', 'candidate_delta'}


def validate_packet(packet):
    if set(packet) != INPUT_KEYS:
        raise ValueError('Only nine legal fields, current base, and current candidate are allowed')
    result = legal_packet({k: v for k, v in packet.items() if k != 'candidate_delta'})
    delta = packet['candidate_delta']
    if (not isinstance(delta, np.ndarray) or delta.dtype != np.float32
            or delta.shape != (1, 32, 14) or not np.isfinite(delta).all()):
        raise ValueError('Finite float32 current candidate required')
    if np.any(delta[:, :, [6, 13]] != 0) or np.max(np.abs(delta)) > np.float32(.1) + 1e-6:
        raise ValueError('Original ARM bound and disabled GRIP required')
    result['candidate_delta'] = delta.copy()
    return result


def at_query(observations, base, raw_delta, query):
    """Extract only the current query from audited actual-history arrays.

    query is a cache index, never a router feature. Raw GRIP proposals are
    excluded because the frozen F candidate never applies them.
    """
    if set(observations) != FORWARD_KEYS or type(query) is not int or query < 0:
        raise ValueError('Audited nine fields and nonnegative query index required')
    arrays = [*observations.values(), base, raw_delta]
    if any(not isinstance(v, np.ndarray) or v.ndim < 1 or len(v) <= query for v in arrays):
        raise ValueError('Current query absent')
    current = {k: v[query:query+1].copy() for k, v in observations.items()}
    proposal = raw_delta[query:query+1]
    if proposal.dtype != np.float32 or proposal.shape != (1, 32, 14):
        raise ValueError('Actual raw native candidate shape/dtype required')
    delta = np.zeros_like(proposal)
    delta[:, :, ARM_INDICES] = proposal[:, :, ARM_INDICES]
    return validate_packet(dict(current, base=base[query:query+1].copy(), candidate_delta=delta))


def concatenate(packets):
    if not packets:
        raise ValueError('Nonempty onset packet collection required')
    checked = [validate_packet(p) for p in packets]
    return {k: np.concatenate([p[k] for p in checked], axis=0) for k in sorted(INPUT_KEYS)}
