from collections.abc import Callable
from typing import Any

from scipy.signal import butter

from espresso.models.ripple_dataset import RippleDataset


class RippleViewerController:
    """Pure data/logic controller. No Qt or PyQt dependencies."""

    def __init__(
        self,
        ripple_datasets: dict[str, RippleDataset],
        spect_low: int = 1,
        spect_high: int = 250,
        z_min: float = -0.5,
        z_max: float = 2.0,
        z_interp: int = 1024,
        nfft: int | None = None,
    ):
        if not ripple_datasets:
            raise ValueError("At least one RippleDataset must be provided")

        self.ripple_datasets = ripple_datasets
        self.dataset_names = list(ripple_datasets.keys())
        self.current_dataset_name: str = self.dataset_names[0]

        first_dataset = ripple_datasets[self.current_dataset_name]
        self.fs = first_dataset.fs

        if self.fs <= 0:
            raise ValueError("Sampling frequency must be positive")

        self.spect_low: int = spect_low
        self.spect_high: int = spect_high
        self.view_window_sec: float = 2.0
        self.nfft: int = int(self.fs * 0.125) if nfft is None else nfft

        self.z_min: float = z_min
        self.z_max: float = z_max
        self.z_interp: int = z_interp

        self.sos = butter(4, [80, 150], btype="band", fs=self.fs, output="sos")
        
        # Track which plots are visible: {dataset_name: {channel: {plot_type: bool}}}
        # plot_type: "raw", "filtered", "hilbert", "spectrogram"
        self.plot_visibility: dict[str, dict[str, dict[str, bool]]] = {}
        self._initialize_plot_visibility()

        self.current_channel: str = self._get_current_channel()
        self.current_ripple_idx: int = 0

        self._listeners: list[Callable[[], None]] = []

    def _initialize_plot_visibility(self) -> None:
        """Initialize visibility state for all dataset/channel/plot combinations."""
        for dataset_name, dataset in self.ripple_datasets.items():
            self.plot_visibility[dataset_name] = {}
            for channel in dataset.raw_volts.keys():
                self.plot_visibility[dataset_name][channel] = {
                    "raw": True,
                    "filtered": False,
                    "hilbert": False,
                    "spectrogram": False,
                }

    def _get_current_channel(self) -> str:
        """Get first available channel from current dataset."""
        current_dataset = self.ripple_datasets[self.current_dataset_name]
        channels = list(current_dataset.raw_volts.keys())
        if not channels:
            raise ValueError(f"Dataset {self.current_dataset_name} has no channels")
        return channels[0]

    @property
    def current_dataset(self) -> RippleDataset:
        return self.ripple_datasets[self.current_dataset_name]

    @property
    def channels(self) -> list[str]:
        return list(self.current_dataset.raw_volts.keys())
    
    def get_dataset(self, dataset_name: str) -> RippleDataset | None:
        """Get a specific dataset by name."""
        return self.ripple_datasets.get(dataset_name)
    
    def get_all_visible_plots(self) -> list[tuple[str, str, str]]:
        """Get all visible plots as (dataset_name, channel, plot_type) tuples."""
        visible_plots = []
        for dataset_name in self.dataset_names:
            for channel in self.ripple_datasets[dataset_name].raw_volts.keys():
                for plot_type in self.get_visible_plots(dataset_name, channel):
                    visible_plots.append((dataset_name, channel, plot_type))
        return visible_plots

    @property
    def n_samples(self) -> int:
        return len(self.current_dataset.raw_volts[self.current_channel])

    @property
    def total_duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def current_ripple_list(self) -> list[Any]:
        return self.current_dataset.ripples.get(self.current_channel, [])

    @property
    def current_ripple(self) -> Any | None:
        ripples = self.current_ripple_list
        if 0 <= self.current_ripple_idx < len(ripples):
            return ripples[self.current_ripple_idx]
        return None

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def notify_listeners(self) -> None:
        for callback in self._listeners:
            callback()

    def change_dataset(self, dataset_name: str) -> None:
        """Switch to a different dataset."""
        if dataset_name not in self.ripple_datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        self.current_dataset_name = dataset_name
        self.current_channel = self._get_current_channel()
        self.current_ripple_idx = 0
        self.notify_listeners()

    def change_channel(self, channel_name: str) -> None:
        if channel_name in self.channels:
            self.current_channel = channel_name
            self.current_ripple_idx = 0
            self.notify_listeners()

    def next_channel(self) -> None:
        current_index = self.channels.index(self.current_channel)
        next_index = min(current_index + 1, len(self.channels) - 1)
        self.change_channel(self.channels[next_index])

    def prev_channel(self) -> None:
        current_index = self.channels.index(self.current_channel)
        prev_index = max(current_index - 1, 0)
        self.change_channel(self.channels[prev_index])

    def next_ripple(self) -> None:
        ripples = self.current_ripple_list
        if ripples and self.current_ripple_idx < len(ripples) - 1:
            self.current_ripple_idx += 1
            self.notify_listeners()

    def prev_ripple(self) -> None:
        ripples = self.current_ripple_list
        if ripples and self.current_ripple_idx > 0:
            self.current_ripple_idx -= 1
            self.notify_listeners()

    def toggle_ripple_highlight(self) -> None:
        """Toggle between 2s and 0.25s view windows."""
        self.view_window_sec = 0.25 if self.view_window_sec >= 0.5 else 2.0
        self.notify_listeners()

    def set_plot_visibility(
        self, dataset_name: str, channel: str, plot_type: str, visible: bool
    ) -> None:
        """Control visibility of a specific plot."""
        if dataset_name not in self.plot_visibility:
            raise ValueError(f"Dataset {dataset_name} not found")
        if channel not in self.plot_visibility[dataset_name]:
            raise ValueError(f"Channel {channel} not found in dataset {dataset_name}")
        if plot_type not in self.plot_visibility[dataset_name][channel]:
            raise ValueError(f"Invalid plot type: {plot_type}")

        self.plot_visibility[dataset_name][channel][plot_type] = visible
        self.notify_listeners()

    def toggle_plot_visibility(
        self, dataset_name: str, channel: str, plot_type: str
    ) -> None:
        """Toggle visibility of a specific plot."""
        current = self.plot_visibility[dataset_name][channel][plot_type]
        self.set_plot_visibility(dataset_name, channel, plot_type, not current)

    def get_plot_visibility(
        self, dataset_name: str, channel: str, plot_type: str
    ) -> bool:
        """Check if a plot is visible."""
        return self.plot_visibility[dataset_name][channel][plot_type]

    def get_visible_plots(self, dataset_name: str, channel: str) -> list[str]:
        """Get list of visible plot types for a dataset/channel."""
        return [
            plot_type
            for plot_type, visible in self.plot_visibility[dataset_name][channel].items()
            if visible
        ]
