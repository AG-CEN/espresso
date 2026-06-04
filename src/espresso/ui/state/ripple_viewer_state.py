from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, TypeAlias

from typing_extensions import Self

from espresso.models.ripple_event import RippleEvent


class PlotType(str, Enum):
    """Supported signal visualization modes."""

    RAW = "raw"
    FILTERED = "filtered"
    HILBERT = "hilbert"
    SPECTROGRAM = "spectrogram"


# Unique identifier of a plot (dataset_name, plot_type).
PlotId: TypeAlias = tuple[str, PlotType]


@dataclass(frozen=True)
class RippleViewerState:
    """Immutable state snapshot representing the exact UI configuration."""

    channel_name: str
    """The name of the currently selected signal channel (e.g., 'Ch1')."""

    ripples: list[RippleEvent]
    """All of the ripples across all of the datasets for the selected channel"""

    plot_visibility: dict[PlotId, bool]
    """Visibility map of plots"""

    current_ripple_index: int = 0
    """The index of the currently focused ripple event."""

    view_window_sec: float = 2.0
    """The temporal width of the chart viewport in seconds (typically 2.0s or 0.25s)."""

    def copy_with(self, **changes: Any) -> Self:
        """
        Returns a new immutable instance with the specified properties updated.
        Mirrors Dart's copyWith pattern natively.
        """
        if "plot_visibility" in changes:
            changes["plot_visibility"] = changes["plot_visibility"].copy()
        return replace(self, **changes)
