import numpy as np
from espresso.models.ripple_event import RippleEvent
from pydantic import BaseModel


class RippleDataset(BaseModel):
    raw_volts: dict[str, np.ndarray]
    ripples: dict[str, list[RippleEvent]]
    fs: float

    model_config = {"arbitrary_types_allowed": True}

    def get_channels(self) -> list[str]:
        return list(self.raw_volts.keys())
