"""Pure signal processing component. No UI dependencies."""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram


class SignalProcessor:
    """Handles all signal processing: filtering, envelope, spectrogram."""

    def __init__(self, fs: float):
        self.fs = fs
        self.sos = butter(4, [80, 150], btype="band", fs=fs, output="sos")

    def process_chunk(
        self, chunk: np.ndarray, nfft: int, spect_low: int, spect_high: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process a signal chunk and return all derivatives.

        Returns:
            (filtered, envelope, freqs, times, spectrogram_data, spec_z)
        """
        filt = sosfiltfilt(self.sos, chunk)
        envelope = np.abs(hilbert(filt))

        nfft = max(1, min(nfft, len(chunk)))
        noverlap = int(nfft * 0.9)
        noverlap = min(noverlap, nfft - 1)

        f, t, sxx = spectrogram(
            chunk, fs=self.fs, nperseg=nfft, noverlap=noverlap, window="hann"
        )

        mask = (f >= spect_low) & (f <= spect_high)
        if np.any(mask):
            s_log = 10 * np.log10(sxx[mask, :] + 1e-12)
            s_z = (s_log - np.mean(s_log, axis=1, keepdims=True)) / (
                np.std(s_log, axis=1, keepdims=True) + 1e-6
            )
        else:
            s_z = np.zeros((0, sxx.shape[1]))

        return filt, envelope, f, t, sxx, s_z
