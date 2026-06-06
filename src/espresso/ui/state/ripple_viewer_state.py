from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, TypeAlias

from typing_extensions import Self

from espresso.models.ripple_event import RippleEvent


class PlotType(str, Enum):
    """Supported signal visualization modes."""

    raw = "raw"
    filtered = "filtered"
    envelope = "envelope"
    spectrogram = "spectrogram"


# Tuple of (dataset label, plot type) makes a unique id for the plot.
PlotId: TypeAlias = tuple[str, PlotType]


class RippleViewerParams:
    spect_low: int = 50
    """Lower frequency limit (Hz) to display in the spectrogram."""

    spect_high: int = 250
    """Upper frequency limit (Hz) to display in the spectrogram."""

    nfft_sec: float = 0.125
    """Duration of the FFT window in seconds. Automatically scaled by sampling rate (fs) to determine window sample size."""

    overlap_ratio: float = 0.95
    """Fractional ratio (0.0 to 1.0) of window overlap. Determines what percentage of the calculated 'nfft' window samples overlap with the next frame."""

    z_min: float = -0.5
    """Minimum standardized power intensity threshold for color mapping baseline."""

    z_max: float = 2.0
    """Maximum standardized power intensity threshold to cap color saturation."""

    z_interp: int = 1024
    """Number of interpolation bins used to smooth the color map rendering gradient."""


@dataclass(frozen=True)
class RippleViewerState:
    """Immutable state snapshot representing the exact UI configuration."""

    channel_name: str
    """The name of the currently selected signal channel (e.g., 'Ch1')."""

    ripples: list[RippleEvent]
    """All of the ripples across all of the datasets for the selected channel"""

    plot_visibility: dict[PlotId, bool]
    """Visibility map of plots."""

    ripple_viewer_params: RippleViewerParams
    """Viewer parameters."""

    current_ripple_index: int = 0
    """The index of the currently focused ripple event."""

    view_window_sec: float = 2.0
    """The temporal width of the chart viewport in seconds (typically 2.0s or 0.25s)."""

    @property
    def current_ripple(self) -> RippleEvent:
        return self.ripples[self.current_ripple_index]

    def copy_with(self, **changes: Any) -> Self:
        """
        Returns a new immutable instance with the specified properties updated.
        Mirrors Dart's copyWith pattern natively.
        """
        if "plot_visibility" in changes:
            changes["plot_visibility"] = changes["plot_visibility"].copy()
        return replace(self, **changes)
