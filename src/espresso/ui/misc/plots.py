from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, spectrogram

class DatasetTrackGroup(QWidget):
    def __init__(self, title: str, fs: float, spect_low: int, spect_high: int, n_samples: int):
        super().__init__()
        self.fs, self.spect_low, self.spect_high, self.n_samples = fs, spect_low, spect_high, n_samples
        self.sos = butter(4, [80, 150], btype="band", fs=self.fs, output="sos")
        self.toggle_map = {}
        self.v_lines = []
        self.init_ui(title)

    def init_ui(self, title):
        self.setMaximumHeight(420)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(1)

        self.p_raw = pg.PlotWidget(title=f"{title} - Raw LFP")
        self.p_filt = pg.PlotWidget(title=f"{title} - Filtered")
        self.p_env = pg.PlotWidget(title=f"{title} - Envelope")
        self.p_spec = pg.PlotWidget(title=f"{title} - Spectrogram")
        self.tracks = {"Raw": self.p_raw, "Filtered": self.p_filt, "Envelope": self.p_env, "Spectrogram": self.p_spec}

        for p in self.tracks.values():
            p.setMaximumHeight(100)
            p.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
            p.setMouseEnabled(y=False)
            p.getPlotItem().setLimits(xMin=0, xMax=self.n_samples / self.fs)
            lay.addWidget(p)
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen((88, 88, 88), width=2, style=pg.QtCore.Qt.PenStyle.DashLine))
            p.addItem(line)
            self.v_lines.append(line)

        self.c_raw = self.p_raw.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_filt = self.p_filt.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_env = self.p_env.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_raw_hi = self.p_raw.plot(pen=pg.mkPen("r", width=2.0))
        self.c_filt_hi = self.p_filt.plot(pen=pg.mkPen("r", width=2.0))
        self.c_env_hi = self.p_env.plot(pen=pg.mkPen("r", width=2.0))

        self.img = pg.ImageItem()
        self.img.setLookupTable(pg.colormap.get("turbo").getLookupTable())
        self.p_spec.addItem(self.img)

    def link_axes(self, master_plot_item):
        for p in self.tracks.values(): p.getPlotItem().setXLink(master_plot_item)

    def update_time_lines(self, peak_sec):
        for line in self.v_lines: line.setPos(peak_sec)

    def refresh_visible_tracks(self):
        any_visible = False
        for name, p in self.tracks.items():
            if self.toggle_map.get(name) and self.toggle_map[name].isChecked():
                p.show()
                any_visible = True
            else:
                p.hide()
        self.show() if any_visible else self.hide()

    def render_track_data(self, s_sec: float, e_sec: float, ds_obj, channel: str):
        s = int(max(0, s_sec * self.fs))
        e = int(min(self.n_samples, e_sec * self.fs))
        if (e - s) < 50 or channel not in ds_obj.raw_volts: return

        x = np.linspace(s / self.fs, (e - 1) / self.fs, e - s)
        chunk = ds_obj.raw_volts[channel][s:e] * 1e6
        f_chunk = sosfiltfilt(self.sos, chunk)
        env_chunk = np.abs(hilbert(f_chunk))

        black_mask = np.ones(chunk.shape, dtype=bool)
        hi_raw, hi_filt, hi_env = [np.full(chunk.shape, np.nan) for _ in range(3)]

        in_view = [r for r in ds_obj.ripples.get(channel, []) if (r.end_sec * self.fs >= s) and (r.start_sec * self.fs <= e)]
        for r in in_view:
            r_s = int(max(s, r.start_sec * self.fs)) - s
            r_e = int(min(e, r.end_sec * self.fs)) - s
            if r_e > r_s:
                r_s_ext, r_e_ext = max(0, r_s - 1), min(len(chunk), r_e + 1)
                hi_raw[r_s_ext:r_e_ext] = chunk[r_s_ext:r_e_ext]
                hi_filt[r_s_ext:r_e_ext] = f_chunk[r_s_ext:r_e_ext]
                hi_env[r_s_ext:r_e_ext] = env_chunk[r_s_ext:r_e_ext]
                black_mask[r_s:r_e] = False

        c_raw_c, c_filt_c, c_env_c = chunk.copy().astype(float), f_chunk.copy().astype(float), env_chunk.copy().astype(float)
        c_raw_c[~black_mask], c_filt_c[~black_mask], c_env_c[~black_mask] = np.nan, np.nan, np.nan

        self.c_raw.setData(x, c_raw_c)
        self.c_filt.setData(x, c_filt_c)
        self.c_env.setData(x, c_env_c)
        self.c_raw_hi.setData(x, hi_raw)
        self.c_filt_hi.setData(x, hi_filt)
        self.c_env_hi.setData(x, hi_env)

        nfft = max(1, min(int(self.fs * 0.125), len(chunk)))
        noverlap = min(int(nfft * 0.9), nfft - 1)
        f, t, sxx = spectrogram(chunk, fs=self.fs, nperseg=nfft, noverlap=noverlap, window="hann")
        mask = (f >= self.spect_low) & (f <= self.spect_high)

        if np.any(mask):
            s_log = 10 * np.log10(sxx[mask, :] + 1e-12)
            s_z = (s_log - np.mean(s_log, axis=1, keepdims=True)) / (np.std(s_log, axis=1, keepdims=True) + 1e-6)
            self.img.setImage(s_z.T, autoLevels=False)
            self.img.setRect(pg.QtCore.QRectF(float(s_sec), float(self.spect_low), float(e_sec - s_sec), float(self.spect_high - self.spect_low)))
            self.p_spec.getPlotItem().setYRange(self.spect_low, self.spect_high, padding=0)

    def sizeHint(self):
        """Forces the declarative framework to reserve clean layout rows."""
        # Provides an explicit baseline pixel size layout mapping hint (Width, Height)
        # 4 internal plots at ~100px each means this container wants around 410px vertically
        return pg.QtCore.QSize(400, 410)
