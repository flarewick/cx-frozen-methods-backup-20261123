"""The gate head and qualified auxiliary head keep their own frozen identities."""
from pathlib import Path
from .contract import GATE_TRAINING,plan
from .paired_heads import build
from .composite_head import CompositeHead

def build_head(training):
    assert Path(training).resolve()==GATE_TRAINING.resolve()
    gate,aux,robots,auxrobots=build()
    return CompositeHead(gate,aux),robots
