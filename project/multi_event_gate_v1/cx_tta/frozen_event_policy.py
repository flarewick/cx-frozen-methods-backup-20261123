"""Zero-correction legacy control with the same base/candidate inference path."""
from .official_onset_domain import OfficialOutputSupportPolicy
from .phase_target_policy import capture_all
from .recovery_contract import rng_capture, rng_fingerprint


class LegacyZeroPolicy(OfficialOutputSupportPolicy):
    def __init__(self, base, training_run, config):
        self.event_config = config
        super().__init__(base, training_run, 'A_bounded_clipped', force_zero=True)
        self.condition = 'Z_zero'
        self.last_event_packet = None

    def predict(self, observation, instruction, noise, *, enabled):
        if enabled is not True: raise ValueError('Common candidate inference required')
        previous = self.previous
        actions, delta, base = super().predict(observation, instruction, noise, enabled=True)
        capture_all(self, previous, base)
        after = rng_fingerprint(rng_capture())
        if after != self.last_diagnostics['rng_after_sampler']:
            raise RuntimeError('Auxiliary inference changed VLA sampler RNG')
        if delta.any() or not (actions == base).all():
            raise RuntimeError('Baseline changed native base actions')
        self.last_diagnostics.update(condition='Z_zero',
            prescription=dict(active=False, strength=0., window=[]),
            gate_training_scope='no_learned_selector', rng_after_gate=after)
        return actions, delta, base
