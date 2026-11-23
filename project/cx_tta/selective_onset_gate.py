"""Conservative execution contract for a future candidate-conditioned router.

No learned weights or calibrated confidence currently exist. This module is
deliberately not wired into the frozen evaluation actor. Bounds must eventually
come from a separately validated predictor/calibrator, not from model entropy
or feature proximity. Tests of this contract are not policy-success evidence.
"""
from dataclasses import dataclass
import numpy as np

from .onset_observables import validate_packet


@dataclass(frozen=True)
class BenefitRiskBounds:
    rescue_lower: float
    rescue_upper: float
    harm_lower: float
    harm_upper: float
    in_support: bool

    def validate(self):
        values = (self.rescue_lower, self.rescue_upper, self.harm_lower, self.harm_upper)
        if not np.isfinite(values).all() or not (
                0 <= self.rescue_lower <= self.rescue_upper <= 1 and
                0 <= self.harm_lower <= self.harm_upper <= 1 and
                self.rescue_lower + self.harm_lower <= 1):
            raise ValueError('Valid marginal probability bounds required')
        if type(self.in_support) is not bool:
            raise ValueError('Explicit support decision required')


class SelectiveOnsetGate:
    """Select Z or exact F at one audited opportunity; consume it even if off.

    Arbitrary later re-entry needs its own paired labels and is not inferred
    from this onset dataset. reset() must be called at each episode boundary.
    There are no default thresholds: they must be frozen outside test scenes.
    """
    def __init__(self, *, harm_cost, harm_budget, gain_margin):
        values = (harm_cost, harm_budget, gain_margin)
        if not np.isfinite(values).all() or harm_cost < 1 or not 0 <= harm_budget <= 1 or gain_margin < 0:
            raise ValueError('Finite conservative utility/risk limits required')
        self.harm_cost = harm_cost
        self.harm_budget = harm_budget
        self.gain_margin = gain_margin
        self.reset()

    def reset(self):
        self.opportunity_consumed = False

    def decide(self, packet, *, opportunity, bounds=None, calibration_accepted=False):
        p = validate_packet(packet)
        if type(opportunity) is not bool or type(calibration_accepted) is not bool:
            raise ValueError('Explicit Boolean opportunity and calibration status')
        result = dict(active=False, reason='outside_audited_opportunity', utility_lower=None)
        if self.opportunity_consumed:
            result['reason'] = 'default_exit'
        elif opportunity:
            self.opportunity_consumed = True
            if not calibration_accepted or bounds is None:
                result['reason'] = 'uncalibrated_abstention'
            else:
                bounds.validate()
                utility = bounds.rescue_lower - self.harm_cost * bounds.harm_upper
                result['utility_lower'] = float(utility)
                if not bounds.in_support:
                    result['reason'] = 'unsupported_abstention'
                elif bounds.harm_upper > self.harm_budget:
                    result['reason'] = 'harm_risk'
                elif utility <= self.gain_margin:
                    result['reason'] = 'insufficient_incremental_benefit'
                else:
                    result.update(active=True, reason='candidate_benefit_with_risk_control')
        action = p['base'].copy()
        if result['active']:
            changed = p['candidate_delta'] != 0
            action[changed] = p['base'][changed] + p['candidate_delta'][changed]
        # Off is byte-exact base; candidate has zero GRIP by packet validation.
        return action, result
