"""Fox — host/symbiont cophylogenetic event-frequency prediction.

Layer-1 delivery API. Minimal real-world path:

    >>> from fox import predict_tgl
    >>> predict_tgl("dataset.tgl")
    {'Speciation': 0.62, 'HGT': 0.10, 'Loss': 0.21, 'Duplication': 0.07}

The input is a well-formatted .tgl bundling the host MSA, symbiont MSA, and
host<->symbiont mapping. No tree inference, no .pt pre-encoding.
"""

from .core import predict_tgl
from .model_loader import load_model
from .encoding import EVENT_NAMES

__all__ = ["predict_tgl", "load_model", "EVENT_NAMES"]
