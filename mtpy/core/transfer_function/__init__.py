import numpy as np

MT_TO_OHM_FACTOR = 1.0 / np.pi * np.sqrt(5.0 / 8.0) * 10**3.5
IMPEDANCE_UNITS = {"mt": 1, "ohm": MT_TO_OHM_FACTOR}

from .z import Z
from .tipper import Tipper
from .pt import PhaseTensor

__all__ = ["Z", "Tipper", "PhaseTensor"]
