from typing import Any, Callable

import numpy as np
from scipy.signal import butter


class RippleViewerController:
    """Pure data/logic controller. No Qt or PyQt dependencies."""

    def __init__(
        self,
        raw_volts: dict[str, np.ndarray],
        ripples: dict[str, list[Any]],  # Replace Any with RippleEvent
        fs: float,
        spect_low: int = 1,
        spect_high: int = 250,
    ):
        if not raw_volts:
            raise ValueError("raw_volts cannot be empty")
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if set(ripples.keys()) - set(raw_volts.keys()):
            raise ValueError("Ripples contain channels not present in raw_volts")

        self.raw = raw_volts
        self.ripples = ripples
        self.fs = fs

        self.channels = list(raw_volts.keys())
        self.current_channel: str = self.channels[0]
        self.current_ripple_idx: int = 0
        self.spect_low: int = spect_low
        self.spect_high: int = spect_high
        self.view_window_sec: float = 2.0
        self.nfft: int = int(self.fs * 0.125)
        self.z_min: float = -0.5
        self.z_max: float = 2.0
        self.z_interp: int = 1024

        self.sos = butter(4, [80, 150], btype="band", fs=self.fs, output="sos")
        self._listeners: list[Callable[[], None]] = []

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def notify_listeners(self) -> None:
        for callback in self._listeners:
            callback()

    @property
    def n_samples(self) -> int:
        return len(self.raw[self.current_channel])

    @property
    def total_duration(self) -> float:
        return self.n_samples / self.fs

    @property
    def current_ripple_list(self) -> list[Any]:
        return self.ripples.get(self.current_channel, [])

    @property
    def current_ripple(self) -> Any | None:
        ripples = self.current_ripple_list
        if 0 <= self.current_ripple_idx < len(ripples):
            return ripples[self.current_ripple_idx]
        return None

    def change_channel(self, channel_name: str) -> None:
        if channel_name in self.channels and channel_name != self.current_channel:
            self.current_channel = channel_name
            self.current_ripple_idx = 0
            self.notify_listeners()

    def next_channel(self) -> None:
        idx = (self.channels.index(self.current_channel) + 1) % len(self.channels)
        self.change_channel(self.channels[idx])

    def prev_channel(self) -> None:
        idx = (self.channels.index(self.current_channel) - 1) % len(self.channels)
        self.change_channel(self.channels[idx])

    def next_ripple(self) -> None:
        ripples = self.current_ripple_list
        if ripples:
            self.current_ripple_idx = (self.current_ripple_idx + 1) % len(ripples)
            self.notify_listeners()

    def prev_ripple(self) -> None:
        ripples = self.current_ripple_list
        if ripples:
            self.current_ripple_idx = (self.current_ripple_idx - 1) % len(ripples)
            self.notify_listeners()

    def update_knobs(
        self, low: int, high: int, nfft: int, z_interp_scaled: int
    ) -> None:
        self.spect_low = low
        self.spect_high = high
        self.nfft = nfft
        self.z_interp = int(z_interp_scaled * 32)
        self.notify_listeners()

    def toggle_ripple_highlight(self) -> None:
        """Toggle between 2s and 0.25s view windows."""
        self.view_window_sec = 0.25 if self.view_window_sec >= 0.5 else 2.0
        self.notify_listeners()
