import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram


class RippleProcessor:
    """Handles signal processing and data preparation for ripple visualization."""

    def __init__(self, fs: float, spect_low: int = 1, spect_high: int = 250):
        """Initialize processor with sampling rate and spectrogram bounds.

        Args:
            fs: Sampling rate in Hz.
            spect_low: Initial low frequency bound in Hz.
            spect_high: Initial high frequency bound in Hz.
        """
        self.fs = fs
        self.spect_low = spect_low
        self.spect_high = spect_high
        self.sos = butter(4, [80, 150], btype="band", fs=self.fs, output="sos")
        self.nfft = int(self.fs * 0.125)
        self.z_interp = 1024
        self.z_min = -0.5
        self.z_max = 2.0

    def process_trace(self, data_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter raw data and compute envelope.

        Args:
            data_chunk: Raw voltage data in microvolts.

        Returns:
            Tuple of (filtered_data, envelope_data).
        """
        filtered = sosfiltfilt(self.sos, data_chunk)
        envelope = np.abs(hilbert(filtered))
        return filtered, envelope

    def compute_ripple_masks(
        self,
        chunk: np.ndarray,
        filtered: np.ndarray,
        envelope: np.ndarray,
        s_idx: int,
        e_idx: int,
        fs: float,
        ripples: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create highlight masks for ripple events in the data chunk.

        Args:
            chunk: Raw data chunk.
            filtered: Filtered data chunk.
            envelope: Envelope data chunk.
            s_idx: Start sample index.
            e_idx: End sample index.
            fs: Sampling rate.
            ripples: List of RippleEvent objects in view.

        Returns:
            Tuple of (black_mask, hi_raw, hi_filt, hi_env) arrays.
        """
        black_mask = np.ones(chunk.shape, dtype=bool)
        hi_raw = np.full(chunk.shape, np.nan)
        hi_filt = np.full(chunk.shape, np.nan)
        hi_env = np.full(chunk.shape, np.nan)

        for ripple in ripples:
            r_s = int(max(s_idx, ripple.start_sec * fs)) - s_idx
            r_e = int(min(e_idx, ripple.end_sec * fs)) - s_idx

            if r_e > r_s:
                r_s_ext = max(0, r_s - 1)
                r_e_ext = min(len(chunk), r_e + 1)

                hi_raw[r_s_ext:r_e_ext] = chunk[r_s_ext:r_e_ext]
                hi_filt[r_s_ext:r_e_ext] = filtered[r_s_ext:r_e_ext]
                hi_env[r_s_ext:r_e_ext] = envelope[r_s_ext:r_e_ext]

                black_mask[r_s:r_e] = False

        return black_mask, hi_raw, hi_filt, hi_env

    def compute_spectrogram(
        self, chunk: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and normalize spectrogram for frequency range.

        Args:
            chunk: Raw data chunk.

        Returns:
            Tuple of (frequencies, times, normalized_spectrogram).
        """
        current_nfft = max(1, min(self.nfft, len(chunk)))
        noverlap = int(current_nfft * 0.9)
        noverlap = min(noverlap, current_nfft - 1)

        f, t, sxx = spectrogram(
            chunk,
            fs=self.fs,
            nperseg=current_nfft,
            noverlap=noverlap,
            window="hann",
        )

        mask = (f >= self.spect_low) & (f <= self.spect_high)
        if not np.any(mask):
            return f, t, np.zeros((np.sum(mask), len(t)))

        s_log = 10 * np.log10(sxx[mask, :] + 1e-12)
        s_z = (s_log - np.mean(s_log, axis=1, keepdims=True)) / (
            np.std(s_log, axis=1, keepdims=True) + 1e-6
        )

        return f[mask], t, s_z

    def get_spectrogram_rect(self, s_sec: float, e_sec: float) -> pg.QtCore.QRectF:
        """Create rectangle for spectrogram positioning.

        Args:
            s_sec: Start time in seconds.
            e_sec: End time in seconds.

        Returns:
            QRectF for spectrogram image placement.
        """
        return pg.QtCore.QRectF(
            float(s_sec),
            float(self.spect_low),
            float(e_sec - s_sec),
            float(self.spect_high - self.spect_low),
        )
