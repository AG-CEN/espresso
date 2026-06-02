"""Plot rendering for signal visualization - extracted from old viewer."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from scipy.signal import hilbert, sosfiltfilt, spectrogram


class SignalPlotRenderer:
    """Renders raw, filtered, envelope, and spectrogram plots."""

    def __init__(
        self,
        p_raw: pg.PlotItem,
        p_filt: pg.PlotItem,
        p_env: pg.PlotItem,
        p_spec: pg.PlotItem,
        win: pg.GraphicsLayoutWidget,
    ):
        self.p_raw = p_raw
        self.p_filt = p_filt
        self.p_env = p_env
        self.p_spec = p_spec
        self.win = win

        # Persistent curve objects
        self.c_raw = self.p_raw.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_filt = self.p_filt.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_env = self.p_env.plot(pen=pg.mkPen((33, 33, 33), width=1))

        self.c_raw_hi = self.p_raw.plot(pen=pg.mkPen("r", width=2.0))
        self.c_filt_hi = self.p_filt.plot(pen=pg.mkPen("r", width=2.0))
        self.c_env_hi = self.p_env.plot(pen=pg.mkPen("r", width=2.0))

        self.img = pg.ImageItem()
        self.img.setLookupTable(pg.colormap.get("turbo").getLookupTable())
        self.p_spec.addItem(self.img)

        self.colorbar = pg.ColorBarItem(values=(-0.5, 2.0), colorMap="turbo")
        self.colorbar.setImageItem(self.img)

        # Vertical lines for ripple markers
        self.v_lines = []
        for p in [self.p_raw, self.p_filt, self.p_env, self.p_spec]:
            line = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen((88, 88, 88), width=2, style=Qt.PenStyle.DashLine),
            )
            p.addItem(line)
            self.v_lines.append(line)

    def render(
        self,
        raw_signal: np.ndarray,
        fs: float,
        sos,
        ripples: list,
        s_sec: float,
        e_sec: float,
        spect_low: int,
        spect_high: int,
        nfft: int,
        z_min: float,
        z_max: float,
    ) -> None:
        """Render all plots - direct translation from old render_all()."""
        s = int(max(0, s_sec * fs))
        e = int(min(len(raw_signal), e_sec * fs))

        if (e - s) < 50:
            return

        x = np.linspace(s / fs, (e - 1) / fs, e - s)
        chunk = raw_signal[s:e] * 1e6
        f_chunk = sosfiltfilt(sos, chunk)
        env_chunk = np.abs(hilbert(f_chunk))

        # Create masks
        black_mask = np.ones(chunk.shape, dtype=bool)
        hi_raw = np.full(chunk.shape, np.nan)
        hi_filt = np.full(chunk.shape, np.nan)
        hi_env = np.full(chunk.shape, np.nan)

        in_view = [
            ripple
            for ripple in ripples
            if (ripple.end_sec * fs >= s) and (ripple.start_sec * fs <= e)
        ]

        for ripple in in_view:
            r_s = int(max(s, ripple.start_sec * fs)) - s
            r_e = int(min(e, ripple.end_sec * fs)) - s

            if r_e > r_s:
                r_s_ext = max(0, r_s - 1)
                r_e_ext = min(len(chunk), r_e + 1)

                hi_raw[r_s_ext:r_e_ext] = chunk[r_s_ext:r_e_ext]
                hi_filt[r_s_ext:r_e_ext] = f_chunk[r_s_ext:r_e_ext]
                hi_env[r_s_ext:r_e_ext] = env_chunk[r_s_ext:r_e_ext]
                black_mask[r_s:r_e] = False

        # Apply masks
        clean_raw = chunk.copy().astype(float)
        clean_filt = f_chunk.copy().astype(float)
        clean_env = env_chunk.copy().astype(float)

        clean_raw[~black_mask] = np.nan
        clean_filt[~black_mask] = np.nan
        clean_env[~black_mask] = np.nan

        # Render curves
        self.c_raw.setData(x, clean_raw)
        self.c_filt.setData(x, clean_filt)
        self.c_env.setData(x, clean_env)

        self.c_raw_hi.setData(x, hi_raw)
        self.c_filt_hi.setData(x, hi_filt)
        self.c_env_hi.setData(x, hi_env)

        # Spectrogram
        nfft = max(1, min(nfft, len(chunk)))
        noverlap = int(nfft * 0.9)
        noverlap = min(noverlap, nfft - 1)
        f, t, sxx = spectrogram(
            chunk, fs=fs, nperseg=nfft, noverlap=noverlap, window="hann"
        )

        mask = (f >= spect_low) & (f <= spect_high)
        if np.any(mask):
            s_log = 10 * np.log10(sxx[mask, :] + 1e-12)
            s_z = (s_log - np.mean(s_log, axis=1, keepdims=True)) / (
                np.std(s_log, axis=1, keepdims=True) + 1e-6
            )

            self.img.setImage(s_z.T, levels=[z_min, z_max])

            self.win.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.win.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            self.img.setRect(
                pg.QtCore.QRectF(
                    float(s_sec),
                    float(spect_low),
                    float(e_sec - s_sec),
                    float(spect_high - spect_low),
                )
            )
            self.p_spec.setYRange(spect_low, spect_high, padding=0)

    def update_ripple_marker(self, ripple_peak_sec: float) -> None:
        """Update vertical lines to ripple peak."""
        for line in self.v_lines:
            line.setPos(ripple_peak_sec)

    def get_colorbar(self) -> pg.ColorBarItem:
        return self.colorbar
