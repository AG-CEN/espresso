import reactivex.operators as ops
from reactivex import Observable
from reactivex.subject import BehaviorSubject

from espresso.models.ripple_dataset import RippleDataset
from espresso.models.ripple_event import RippleEvent
from espresso.ui.state.ripple_viewer_state import (
    PlotId,
    PlotType,
    RippleViewerParams,
    RippleViewerState,
)


class RippleViewerController:
    """Controller for managing viewer state and processing ripple event data."""

    def __init__(
        self,
        ripple_datasets: list[RippleDataset],
        ripple_viewer_params: RippleViewerParams,
    ):
        if not ripple_datasets:
            raise ValueError("At least one RippleDataset must be provided")

        self.ripple_datasets: list[RippleDataset] = ripple_datasets
        initial_state = RippleViewerState(
            channel_name=self.channels[0],
            ripples=self._calculate_current_ripples(
                channel_name=list(ripple_datasets[0].ripples.keys())[0]
            ),
            current_ripple_index=0,
            view_window_sec=2.0,
            plot_visibility={
                (dataset.label, plot_type): True
                for dataset in self.ripple_datasets
                for plot_type in PlotType
            },
            ripple_viewer_params=ripple_viewer_params,
        )

        self._state_subject = BehaviorSubject(value=initial_state)

    @property
    def state(self) -> RippleViewerState:
        return self._state_subject.value

    @state.setter
    def _state(self, new_state: RippleViewerState) -> None:
        self._state_subject.on_next(new_state)

    @property
    def stream(self) -> Observable[RippleViewerState]:
        return self._state_subject.pipe(ops.as_observable())

    @property
    def channels(self) -> list[str]:
        return list(next(iter(self.ripple_datasets)).raw_microvolts.keys())

    def _calculate_current_ripples(
        self, channel_name: str
    ) -> list[tuple[str, RippleEvent]]:
        flattened_ripples = [
            (dataset.label, ripple)
            for dataset in self.ripple_datasets
            for ripple in dataset.ripples[channel_name]
        ]

        sorted_ripples = sorted(flattened_ripples, key=lambda r: r[1].peak_sec)

        return sorted_ripples

    def change_channel(self, channel_name: str) -> None:
        """Change current active viewing channel."""
        if channel_name in self.channels:
            self._state = self.state.copy_with(
                channel_name=channel_name,
                current_ripple_index=0,
                ripples=self._calculate_current_ripples(channel_name=channel_name),
            )

    def next_channel(self) -> None:
        """Advance viewport to the next channel sequence."""
        current_index = self.channels.index(self._state.channel_name)
        next_index = min(current_index + 1, len(self.channels) - 1)
        self.change_channel(self.channels[next_index])

    def prev_channel(self) -> None:
        """Regress viewport to the previous channel sequence."""
        current_index = self.channels.index(self._state.channel_name)
        prev_index = max(current_index - 1, 0)
        self.change_channel(self.channels[prev_index])

    def next_ripple(self) -> None:
        """Focus on the next ripple."""
        ripples = self._state.ripples
        current_index = self._state.current_ripple_index
        if ripples and current_index < len(ripples) - 1:
            self._state = self.state.copy_with(
                current_ripple_index=current_index + 1,
            )

    def prev_ripple(self) -> None:
        """Focus on the previous ripple."""
        ripples = self._state.ripples
        current_index = self._state.current_ripple_index
        if ripples and current_index > 0:
            self._state = self.state.copy_with(
                current_ripple_index=current_index - 1,
            )

    def on_plot_visibility_toggled(
        self,
        plot_id: PlotId,
        new_value: bool,
    ) -> None:
        self._state = self.state.copy_with(
            plot_visibility={**self._state.plot_visibility, plot_id: new_value}
        )

    def toggle_ripple_highlight(self) -> None:
        """Toggle between zoomed-in (0.25s) and panoramic (2.0s) view windows."""
        new_V = 0.25 if self._state.view_window_sec >= 0.5 else 2.0
        self._state = self.state.copy_with(view_window_sec=new_V)
