import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QGraphicsGridLayout
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram

from espresso.models.ripple_dataset import RippleDataset
from espresso.models.ripple_event import RippleEvent
from espresso.ui.state.ripple_viewer_state import PlotType, RippleViewerState


class PlotsView(pg.GraphicsLayoutWidget):
    def __init__(
        self,
        initial_state: RippleViewerState,
        ripple_dataset: RippleDataset,
    ):
        super().__init__()
        self.ripple_dataset = ripple_dataset
        self._init_ui(initial_state=initial_state)
        self._focus_on_ripple(initial_state.current_ripple)

    def _init_ui(self, initial_state: RippleViewerState):
        self._last_state = initial_state
        self.sos = butter(
            4, [80, 150], btype="band", fs=self.ripple_dataset.fs, output="sos"
        )
        pg.setConfigOptions(useOpenGL=True, antialias=True)

        self._create_raw_plot()
        self._create_filtered_plot()
        self._create_envelope_plot()
        self._create_spectrogram()

        self.all_plots: list[pg.PlotItem] = [
            self.p_raw,
            self.p_filt,
            self.p_env,
            self.p_spec,
        ]
        self.v_lines: list[pg.InfiniteLine] = []
        for plot in self.all_plots:
            line = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen((88, 88, 88), width=2, style=Qt.PenStyle.DashLine),
            )
            self._set_plot_defaults(plot=plot)
            plot.addItem(line)
            self.v_lines.append(line)

        self.p_raw.sigRangeChanged.connect(
            lambda vb, range_list: self.build(ripple_viewer_state=self._last_state)
        )

    def _create_raw_plot(self):
        self.p_raw = self.ci.addPlot(row=0, col=0)
        self.p_raw.setLabel("left", "Raw", units="V")
        self.c_raw = self.p_raw.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_raw_hi = self.p_raw.plot(pen=pg.mkPen("r", width=2.0))

    def _create_filtered_plot(self):

        self.p_filt = self.ci.addPlot(row=1, col=0)
        self.p_filt.setLabel("left", "Filtered", units="V")
        self.c_filt = self.p_filt.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_filt_hi = self.p_filt.plot(pen=pg.mkPen("r", width=2.0))

    def _create_envelope_plot(self):
        self.p_env = self.ci.addPlot(row=2, col=0)
        self.p_env.setLabel("left", "Envelope", units="V")
        self.c_env = self.p_env.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_env_hi = self.p_env.plot(pen=pg.mkPen("r", width=2.0))

    def _create_spectrogram(self):
        self.p_spec = self.ci.addPlot(row=3, col=0)
        self.p_spec.setLabel("bottom", "Time", units="s")
        self.img = pg.ImageItem()
        self.img.setLookupTable(pg.colormap.get("turbo").getLookupTable())
        self.p_spec.addItem(self.img)
        colorbar = pg.ColorBarItem(
            values=(-0.5, 2.0),
            colorMap="turbo",
            width=20,
        )
        colorbar.setMaximumWidth(20)
        colorbar.setImageItem(self.img)

        plot_layout = self.p_spec.layout
        if isinstance(plot_layout, QGraphicsGridLayout):
            plot_layout.addItem(colorbar, 2, 3)
            plot_layout.setColumnFixedWidth(3, 20)

    def _set_plot_defaults(
        self,
        plot: pg.PlotItem,
        grid_color="k",
    ) -> None:
        plot.showGrid(x=True, y=True, alpha=0.5)
        plot.getViewBox().setMouseEnabled(y=False)
        grid_pen = pg.mkPen(color=grid_color, width=1)
        plot.getAxis("bottom").setPen(grid_pen)
        plot.getAxis("left").setPen(grid_pen)

    def set_x_link(self, reference_plot: pg.PlotItem):
        self.p_raw.getViewBox().setXLink(reference_plot)
        self.p_filt.getViewBox().setXLink(reference_plot)
        self.p_env.getViewBox().setXLink(reference_plot)
        self.p_spec.getViewBox().setXLink(reference_plot)

    def update_ripple_marker(self, ripple_peak_sec: float) -> None:
        for line in self.v_lines:
            line.setPos(ripple_peak_sec)

    def _focus_on_ripple(self, ripple: RippleEvent):
        for line in self.v_lines:
            line.setPos(ripple.peak_sec)

        center = ripple.peak_sec
        half_window = self._last_state.view_window_sec / 2

        self.p_raw.getViewBox().setXRange(
            center - half_window,
            center + half_window,
            padding=0,
        )

    def _handle_state(self, new_state: RippleViewerState):
        has_ripple_changed = (
            new_state.current_ripple_index != self._last_state.current_ripple_index
            or new_state.channel_name != self._last_state.channel_name
        )
        if has_ripple_changed:
            self._focus_on_ripple(ripple=new_state.current_ripple)

        has_zoom_changed = new_state.view_window_sec != self._last_state.view_window_sec
        if has_zoom_changed:
            x_range, _ = self.p_raw.getViewBox().viewRange()
            center = (x_range[0] + x_range[1]) / 2
            self.p_raw.getViewBox().setXRange(
                center - new_state.view_window_sec / 2,
                center + new_state.view_window_sec / 2,
                padding=0,
            )

        self.p_raw.setVisible(
            new_state.plot_visibility[(self.ripple_dataset.label, PlotType.raw)]
        )
        self.p_filt.setVisible(
            new_state.plot_visibility[(self.ripple_dataset.label, PlotType.filtered)]
        )
        self.p_env.setVisible(
            new_state.plot_visibility[(self.ripple_dataset.label, PlotType.envelope)]
        )
        self.p_spec.setVisible(
            new_state.plot_visibility[(self.ripple_dataset.label, PlotType.spectrogram)]
        )

    def build(self, ripple_viewer_state: RippleViewerState) -> None:
        """Render plots for a dataset."""

        self._handle_state(new_state=ripple_viewer_state)
        self._last_state = ripple_viewer_state

        dataset = self.ripple_dataset

        raw_signal = dataset.raw_volts[ripple_viewer_state.channel_name]
        fs = dataset.fs
        n_samples = len(raw_signal)

        for plot in self.all_plots:
            plot.getViewBox().setLimits(
                xMin=0,
                xMax=n_samples / fs,
                maxXRange=ripple_viewer_state.view_window_sec,
            )

        vr = self.p_raw.getViewBox().viewRange()
        s_sec, e_sec = vr[0][0], vr[0][1]

        s = int(max(0, s_sec * fs))
        e = int(min(n_samples, e_sec * fs))

        if (e - s) < 50:
            return

        x = np.linspace(s / dataset.fs, (e - 1) / dataset.fs, e - s)
        chunk = dataset.raw_volts[ripple_viewer_state.channel_name][s:e] / 1e6
        f_chunk = sosfiltfilt(self.sos, chunk)
        env_chunk = np.abs(hilbert(f_chunk))

        # 1. Create Masks
        # Default everything to visible (Black)
        black_mask = np.ones(chunk.shape, dtype=bool)
        # Default highlights to empty (Red)
        hi_raw = np.full(chunk.shape, np.nan)
        hi_filt = np.full(chunk.shape, np.nan)
        hi_env = np.full(chunk.shape, np.nan)

        # Find ripples in view
        in_view = [
            ripple
            for ripple in dataset.ripples[ripple_viewer_state.channel_name]
            if (ripple.end_sec * dataset.fs >= s)
            and (ripple.start_sec * dataset.fs <= e)
        ]

        for ripple in in_view:
            # Convert seconds to samples and clip to the current view [s, e]
            r_s = int(max(s, ripple.start_sec * dataset.fs)) - s
            r_e = int(min(e, ripple.end_sec * dataset.fs)) - s

            if r_e > r_s:
                # 1. Expand red indices by 1 to overlap with black line
                # Clamp to 0 and len(chunk) to avoid index errors
                r_s_ext = max(0, r_s - 1)
                r_e_ext = min(len(chunk), r_e + 1)

                # 2. Transfer data to Red arrays using the EXTENDED range
                hi_raw[r_s_ext:r_e_ext] = chunk[r_s_ext:r_e_ext]
                hi_filt[r_s_ext:r_e_ext] = f_chunk[r_s_ext:r_e_ext]
                hi_env[r_s_ext:r_e_ext] = env_chunk[r_s_ext:r_e_ext]

                # 3. Mask the Black line using the ORIGINAL range
                # This keeps the black line's boundary sample visible
                black_mask[r_s:r_e] = False

        # 2. Apply the "Cut" to Black signals
        # We create a copy to avoid modifying the original data buffers
        clean_raw = chunk.copy().astype(float)
        clean_filt = f_chunk.copy().astype(float)
        clean_env = env_chunk.copy().astype(float)

        clean_raw[~black_mask] = np.nan
        clean_filt[~black_mask] = np.nan
        clean_env[~black_mask] = np.nan

        # 3. Render
        self.c_raw.setData(x, clean_raw)
        self.c_filt.setData(x, clean_filt)
        self.c_env.setData(x, clean_env)

        self.c_raw_hi.setData(x, hi_raw)
        self.c_filt_hi.setData(x, hi_filt)
        self.c_env_hi.setData(x, hi_env)

        # 2. Update Spectrogram

        nfft: int = max(1, min(dataset.viewer_params.nfft, len(chunk)))
        n_overlap = int(nfft * 0.9)
        n_overlap: int = min(n_overlap, nfft - 1)
        f, t, sxx = spectrogram(
            chunk, fs=dataset.fs, nperseg=nfft, noverlap=n_overlap, window="hann"
        )

        mask = (f >= dataset.viewer_params.spect_low) & (
            f <= dataset.viewer_params.spect_high
        )

        if np.any(mask):
            s_log = 10 * np.log10(sxx[mask, :] + 1e-12)
            s_z = (s_log - np.mean(s_log, axis=1, keepdims=True)) / (
                np.std(s_log, axis=1, keepdims=True) + 1e-6
            )

            self.img.setImage(
                s_z.T, levels=[dataset.viewer_params.z_min, dataset.viewer_params.z_max]
            )

            self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            self.img.setRect(
                pg.QtCore.QRectF(
                    float(s_sec),
                    float(dataset.viewer_params.spect_low),
                    float(e_sec - s_sec),
                    float(
                        dataset.viewer_params.spect_high
                        - dataset.viewer_params.spect_low
                    ),
                )
            )
            self.p_spec.getViewBox().setYRange(
                dataset.viewer_params.spect_low,
                dataset.viewer_params.spect_high,
                padding=0,
            )
