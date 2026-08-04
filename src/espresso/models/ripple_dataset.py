from dataclasses import dataclass

import numpy as np

from espresso.models.ripple_event import RippleEvent


@dataclass
class RippleDataset:
    label: str
    raw_microvolts: dict[str, np.ndarray]
    ripples: dict[str, list[RippleEvent]]
    fs: float
    bandpass_filter: tuple[int, int]
