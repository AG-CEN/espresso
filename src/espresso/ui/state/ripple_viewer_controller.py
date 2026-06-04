import heapq
from collections.abc import Iterator
from typing import Any, TypeAlias

import reactivex.operators as ops
from reactivex import Observable
from reactivex.subject import BehaviorSubject

from espresso.models.ripple_dataset import RippleDataset
from espresso.models.ripple_event import RippleEvent
from espresso.ui.state.ripple_viewer_state import PlotId, PlotType, RippleViewerState


class RippleViewerController:
    """Controller for managing viewer state and processing ripple event data."""

    def __init__(
        self,
        ripple_datasets: dict[str, RippleDataset],
    ):
        if not ripple_datasets:
            raise ValueError("At least one RippleDataset must be provided")

        self.ripple_datasets: dict[str, RippleDataset] = ripple_datasets

        initial_state = RippleViewerState(
            channel_name=self.channels[0],
            current_ripple_index=0,
            view_window_sec=2.0,
            plot_visibility={
                (dataset_name, plot_type): True
                for dataset_name in self.ripple_datasets
                for plot_type in PlotType
            },
        )

        self._state_subject = BehaviorSubject(value=initial_state)

    @property
    def _state(self) -> RippleViewerState:
        return self._state_subject.value

    @_state.setter
    def _state(self, new_state: RippleViewerState) -> None:
        self._state_subject.on_next(new_state)

    @property
    def stream(self) -> Observable[RippleViewerState]:
        return self._state_subject.pipe(ops.as_observable())

    @property
    def channels(self) -> list[str]:
        return list(next(iter(self.ripple_datasets.values())).raw_volts.keys())

    @property
    def _current_ripples(self) -> list[RippleEvent]:
        """Merge sorted channel data and remove duplicates within 50ms."""
        streams: list[list[RippleEvent]] = [
            d.ripples[self.current_channel]
            for d in self.ripple_datasets.values()
            if self.current_channel in d.ripples
        ]
        sorted_ripples: Iterator[RippleEvent] = heapq.merge(
            *streams, key=lambda r: r.peak_sec
        )

        filtered: list[RippleEvent] = []
        for ripple in sorted_ripples:
            if not filtered or (ripple.peak_sec - filtered[-1].peak_sec >= 0.05):
                filtered.append(ripple)
        return filtered

    def change_channel(self, channel_name: str) -> None:
        """Change current active viewing channel."""
        if channel_name in self.channels:
            self._state = self._state.copy_with(
                current_channel=channel_name,
                current_ripple_idx=0,
            )

    def on_plot_visibility_toggled(
        self,
        dataset_name: str,
        plot_type: PlotType,
        new_value: bool,
    ) -> None:
        # TODO implement
        print("not implemented")

    def next_channel(self) -> None:
        """Advance viewport to the next channel sequence."""
        current_index = self.channels.index(self.current_channel)
        next_index = min(current_index + 1, len(self.channels) - 1)
        self.change_channel(self.channels[next_index])

    def prev_channel(self) -> None:
        """Regress viewport to the previous channel sequence."""
        current_index = self.channels.index(self.current_channel)
        prev_index = max(current_index - 1, 0)
        self.change_channel(self.channels[prev_index])

    def next_ripple(self) -> None:
        """Focus on the next chronological ripple event in the sequence."""
        ripples = self.current_ripples
        if ripples and self.current_ripple_idx < len(ripples) - 1:
            self.current_ripple_idx += 1
            self.notify_listeners()

    def prev_ripple(self) -> None:
        """Focus on the previous chronological ripple event in the sequence."""
        ripples = self.current_ripples
        if ripples and self.current_ripple_idx > 0:
            self.current_ripple_idx -= 1
            self.notify_listeners()

    def toggle_ripple_highlight(self) -> None:
        """Toggle between zoomed-in (0.25s) and panoramic (2.0s) view windows."""
        self.view_window_sec = 0.25 if self.view_window_sec >= 0.5 else 2.0
        self.notify_listeners()
