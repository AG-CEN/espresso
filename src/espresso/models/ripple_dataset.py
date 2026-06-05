from dataclasses import dataclass, field

import numpy as np

from espresso.models.ripple_event import RippleEvent


@dataclass
class RippleViewerParams:
    spect_low: int = 50
    spect_high: int = 250
    nfft: int = field(init=False)
    z_min: float = -0.5
    z_max: float = 2.0
    z_interp: int = 1024


@dataclass
class RippleDataset:
    label: str
    raw_volts: dict[str, np.ndarray]
    ripples: dict[str, list[RippleEvent]]
    fs: float

    viewer_params: RippleViewerParams = field(default_factory=RippleViewerParams)

    def __post_init__(self) -> None:
        self.viewer_params.nfft = int(self.fs * 0.125)
