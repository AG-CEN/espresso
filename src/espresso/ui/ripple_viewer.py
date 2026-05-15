import sys
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QDial,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram

from espresso.models.ripple_event import RippleEvent


class RippleViewer(QWidget):
    def __init__(
        self,
        raw_volts: dict[str, np.ndarray],
        ripples: dict[str, list[RippleEvent]],
        fs: float,
        spect_low: int = 1,
        spect_high: int = 250,
    ):
        """Visualization of neural signals and detected ripple events.

        Plots raw voltage, filtered band, envelope, spectrograms, and event bounds per
        channel.

        Args:
            raw_volts (dict[str, np.ndarray]): Raw voltage arrays per channel name.
            ripples (dict[str, list[RippleEvent]]): RippleEvent lists per channel name.
            fs (float): Sampling rate in Hz.
            spect_low (int, optional): Initial low frequency bound in Hz. Defaults to 1.
            spect_high (int, optional): Initial high frequency bound in Hz. Defaults to 250.
        """  # noqa: E501

        self.app: QCoreApplication | None = QApplication.instance()
        if self.app is None:
            # Instantiate a fresh app context if none is alive
            self.app = QApplication(sys.argv)
            self._owns_app = True
        else:
            self._owns_app = False

        if not raw_volts:
            raise ValueError('raw_volts cannot be empty')
        if fs <= 0:
            raise ValueError('Sampling frequency must be positive')

        if set(ripples.keys()) - set(raw_volts.keys()):
            raise ValueError('Ripples contain channels not present in raw_volts')

        super().__init__()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(useOpenGL=True)

        self.raw: dict[str, np.ndarray[tuple[Any, ...], np.dtype[Any]]] = raw_volts
        self.ripples: dict[str, list[RippleEvent]] = ripples
        self.fs: float = fs

        self.spect_low: int = spect_low
        self.spect_high: int = spect_high

        self.current_channel: str = list(raw_volts.keys())[0]
        self.n_samples = len(raw_volts[self.current_channel])
        self.sos = butter(4, [80, 150], btype='band', fs=self.fs, output='sos')
        self.current_ripple: int = 0
        self.view_window_sec = 2.0
        self.nfft = int(self.fs * 0.125)
        self.z_min = -0.5
        self.z_max = 2.0
        self.z_interp = 1024
        self.knob_labels = {}
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 1. Top Navigation Bar (Buttons)
        top_layout = QHBoxLayout()

        # --- Left: Channel Buttons ---
        self.prev_ch_btn = QPushButton('Ch -')
        self.next_ch_btn = QPushButton('Ch +')
        self.ch_input = QLineEdit(self.current_channel)
        self.ch_input.setFixedWidth(80)
        self.ch_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ch_input.setStyleSheet("""
            QLineEdit {
                font-weight: bold; 
                font-size: 14px; 
                border: 1px solid #999; 
                border-radius: 4px;
                padding: 2px;
            }
        """)
        # --- Middle: Info Label ---
        self.info_label = QLabel('0/0')
        self.info_label.setStyleSheet('font-weight: bold; font-size: 14px;')

        # --- Right: Ripple Buttons ---
        self.prev_btn = QPushButton('<')
        self.next_btn = QPushButton('>')

        # Styling
        all_btns = [self.prev_ch_btn, self.next_ch_btn, self.prev_btn, self.next_btn]
        for b in all_btns:
            width = 60 if 'Ch' in b.text() else 40
            b.setFixedSize(width, 30)
            b.setStyleSheet(
                'font-weight: bold; border: 1px solid #999; border-radius: 4px;'
            )

        # Layout Assembly
        # Left
        top_layout.addWidget(self.prev_ch_btn)
        top_layout.addWidget(self.ch_input)
        top_layout.addWidget(self.next_ch_btn)

        top_layout.addStretch()  # Pushes info to the center

        # Middle
        top_layout.addWidget(self.info_label, alignment=Qt.AlignmentFlag.AlignCenter)

        top_layout.addStretch()  # Pushes ripple buttons to the right

        # Right
        top_layout.addWidget(self.prev_btn)
        top_layout.addWidget(self.next_btn)

        layout.addLayout(top_layout)
        self.ch_input.clearFocus()

        # 2. Main Graphics Window (Plots)
        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        pg.setConfigOptions(useOpenGL=True, antialias=True)

        self.p_raw = self._add_grilled_plot(0, 'Raw LFP')
        self.p_filt = self._add_grilled_plot(1, 'Filtered')
        self.p_env = self._add_grilled_plot(2, 'Envelope')
        self.p_spec = self._add_grilled_plot(
            3, 'Spectrogram', grid_color=(190, 190, 190)
        )

        self.c_raw_hi = self.p_raw.plot(pen=pg.mkPen('r', width=2.0))
        self.c_filt_hi = self.p_filt.plot(pen=pg.mkPen('r', width=2.0))
        self.c_env_hi = self.p_env.plot(pen=pg.mkPen('r', width=2.0))

        self.p_raw.setLabel('left', 'Voltage', units='µV')
        self.p_filt.setLabel('left', 'Filtered', units='µV')
        self.p_env.setLabel('left', 'Envelope', units='µV')
        self.p_spec.setLabel('bottom', 'Time', units='s')
        self.p_raw.setLabel('left', 'Voltage', units='µV')

        # 3. Knob Section (Between Spec and Nav)
        self.knob_layout = QGridLayout()
        self.k_low = self._add_knob('Low Hz', 1, 249, self.spect_low, 0)
        self.k_high = self._add_knob(
            'High Hz', 1, int(self.fs // 2) - 1, self.spect_high, 1
        )
        self.k_nfft = self._add_knob('NFFT', 1, int(self.fs * 0.5), self.nfft, 2)

        self.k_interp = self._add_knob(
            'Z-Interp', 1, 100, int(self.z_interp // 32), col=3, single_step=1
        )

        knob_widget = QWidget()
        knob_widget.setLayout(self.knob_layout)
        knob_widget.setMaximumHeight(90)
        knob_widget.setStyleSheet('background-color: transparent;')
        knob_proxy = QGraphicsProxyWidget()
        knob_proxy.setMinimumHeight(90)
        knob_proxy.setWidget(knob_widget)
        self.win.addItem(knob_proxy, row=4, col=0)  # ty:ignore[unknown-argument]

        # 4. Bottom Navigation View (Fixed World View)
        self.p_nav = self.win.addPlot(row=5, col=0)  # ty:ignore[unresolved-attribute]
        self.p_nav.setMaximumHeight(40)
        self.p_nav.setMouseEnabled(y=False)
        self.p_nav.hideAxis('left')
        self.p_nav.hideAxis('bottom')
        total_duration = self.n_samples / self.fs
        self.p_nav.setLimits(xMin=0, xMax=total_duration, minXRange=total_duration)
        self.p_nav.getViewBox().setBackgroundColor(pg.mkColor(100, 100, 100, 25))

        # Pre-decimated background signal for nav bar
        dec = max(500, self.n_samples // 5000)
        nav_x = np.arange(0, self.n_samples, dec) / self.fs
        self.p_nav.plot(nav_x, self.raw[self.current_channel][::dec], pen='k')

        # Draw ripples as horizontal bars
        if self.ripples[self.current_channel]:
            for ripple in self.ripples[self.current_channel]:
                start_sec = ripple.start_sec
                end_sec = ripple.end_sec
                width = max(0.001, end_sec - start_sec)
                item = QGraphicsRectItem(start_sec, -1, width, 2)
                item.setPen(pg.mkPen('r', width=0.5))
                item.setBrush(pg.mkBrush(255, 0, 0, 100))

                self.p_nav.addItem(item)

        # Navigation draggable slider
        self.nav_line = pg.InfiniteLine(
            pos=0,
            movable=False,  # Disable dragging
            pen=pg.mkPen('r', width=2),
        )
        self.nav_line.setAcceptHoverEvents(False)
        self.p_nav.addItem(self.nav_line)

        # 5. Persistent Curve Objects (Prevent memory leaks and flickering)
        self.c_raw = self.p_raw.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_filt = self.p_filt.plot(pen=pg.mkPen((33, 33, 33), width=1))
        self.c_env = self.p_env.plot(pen=pg.mkPen((33, 33, 33), width=1))

        self.img = pg.ImageItem()
        self.img.setLookupTable(pg.colormap.get('turbo').getLookupTable())
        self.p_spec.addItem(self.img)

        # Color Bar for the spectogram
        self.colorbar = pg.ColorBarItem(
            values=(self.z_min, self.z_max), colorMap='turbo'
        )
        self.colorbar.setImageItem(self.img)
        self.win.addItem(self.colorbar, 3, 1)

        self.v_lines = []
        for p in [self.p_raw, self.p_filt, self.p_env, self.p_spec]:
            line = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen((88, 88, 88), width=2, style=Qt.PenStyle.DashLine),
            )
            p.addItem(line)
            self.v_lines.append(line)

        # Connect Signals
        self.prev_btn.clicked.connect(self.go_prev_ripple)
        self.next_btn.clicked.connect(self.go_next_ripple)
        self.prev_ch_btn.clicked.connect(self.go_prev_channel)
        self.next_ch_btn.clicked.connect(self.go_next_channel)
        self.ch_input.returnPressed.connect(self._go_to_channel_from_input)
        self.p_nav.scene().sigMouseClicked.connect(self._on_nav_clicked)
        self.p_raw.sigRangeChanged.connect(self._on_plot_moved)
        self.nav_line.sigPositionChanged.connect(self._sync_nav_line_to_view)

        # Initial jump to first detected ripple
        self.go_to_ripple(0)
        self.render_all()

    def _on_plot_moved(self) -> None:
        view_range = self.p_raw.viewRange()[0]
        center_sec = (view_range[0] + view_range[1]) / 2
        self.nav_line.blockSignals(True)
        self.nav_line.setValue(center_sec)
        self.nav_line.blockSignals(False)
        self.render_all()

    def _add_grilled_plot(self, row, title, grid_color='k'):
        p = self.win.addPlot(row=row, col=0, title=title)  # ty:ignore[unresolved-attribute]
        p.showGrid(x=True, y=True, alpha=0.5)
        p.setMouseEnabled(y=False)
        grid_pen = pg.mkPen(color=grid_color, width=1)

        p.getAxis('bottom').setPen(grid_pen)
        p.getAxis('left').setPen(grid_pen)
        # Lock by capping horizontal zoom out
        p.setLimits(
            xMin=0, xMax=self.n_samples / self.fs, maxXRange=self.view_window_sec
        )
        if row > 0:
            p.setXLink(self.p_raw)
        return p

    def _add_knob(self, label, min_v, max_v, cur_v, col, single_step=10):
        k = QDial()
        k.setFixedSize(48, 48)
        k.setRange(min_v, max_v)
        k.setValue(cur_v)
        k.setSingleStep(single_step)
        k.setNotchesVisible(True)
        k.valueChanged.connect(self._on_knob_changed)
        val_lbl = QLabel(f'{label}: {cur_v}')
        val_lbl.setStyleSheet('font-size: 14px; color: #666;')
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.knob_labels[id(k)] = val_lbl
        self.knob_layout.addWidget(k, 1, col, alignment=Qt.AlignmentFlag.AlignCenter)
        self.knob_layout.addWidget(val_lbl, 2, col)
        self.knob_layout.setColumnStretch(col, 1)

        return k

    def _update_knob_labels(self) -> None:
        self._update_knob_label(self.k_low, self.spect_low)
        self._update_knob_label(self.k_high, self.spect_high)
        self._update_knob_label(self.k_nfft, self.nfft)
        self._update_knob_label(self.k_interp, self.z_interp)

    def _update_knob_label(self, knob, new_value):
        knob_label = self.knob_labels[id(knob)]
        if knob_label:
            prefix = knob_label.text().split(':')[0]
            knob_label.setText(f'{prefix}: {new_value}')

    def _on_knob_changed(self) -> None:
        self.spect_low = self.k_low.value()
        self.spect_high = self.k_high.value()
        self.nfft = self.k_nfft.value()
        self.z_interp = self.k_interp.value() * 32
        self._update_knob_labels()
        # TODO(ben): debounce this to prevent excessive rendering during knob adjustment
        self.render_all()

    def _sync_nav_line_to_view(self) -> None:
        center = self.nav_line.value()
        half_window = self.view_window_sec / 2
        self.p_raw.setXRange(center - half_window, center + half_window, padding=0)

    def _on_nav_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.p_nav.sceneBoundingRect().contains(pos):
                mouse_point = self.p_nav.vb.mapSceneToView(pos)
                new_time = mouse_point.x()
                self.nav_line.setValue(new_time)
                self.render_all()

    def go_to_ripple(self, idx):
        if not self.ripples[self.current_channel]:
            return
        self.current_ripple = np.clip(
            idx, 0, len(self.ripples[self.current_channel]) - 1
        )
        ripple = self.ripples[self.current_channel][self.current_ripple]

        for line in self.v_lines:
            line.setPos(ripple.peak_sec)

        self.nav_line.setValue(ripple.peak_sec)

        self.info_label.setText(
            f"""RIPPLE 
            {self.current_ripple + 1} / {len(self.ripples[self.current_channel])}"""
        )

    def go_next_channel(self) -> None:
        channels = list(self.raw.keys())
        current_idx = channels.index(self.current_channel)
        self._go_to_channel(current_idx + 1)

    def go_prev_channel(self) -> None:
        channels = list(self.raw.keys())
        current_idx = channels.index(self.current_channel)
        self._go_to_channel(current_idx - 1)

    def _go_to_channel(self, idx) -> None:
        channels = list(self.raw.keys())
        self.current_channel = channels[np.clip(idx, 0, len(channels) - 1)]
        self.ch_input.setText(self.current_channel)
        self.go_to_ripple(0)
        self.render_all()

    def _go_to_channel_from_input(self) -> None:
        channel_name = self.ch_input.text()
        self.ch_input.clearFocus()
        if channel_name in self.raw:
            self.current_channel = channel_name
            self.go_to_ripple(0)
            self.render_all()
        else:
            self.ch_input.setText(self.current_channel)

    def go_next_ripple(self) -> None:
        self.go_to_ripple(self.current_ripple + 1)

    def go_prev_ripple(self) -> None:
        self.go_to_ripple(self.current_ripple - 1)

    def toggle_ripple_highlight(self) -> None:
        if self.view_window_sec < 0.5:
            self.view_window_sec = 2.0
        else:
            self.view_window_sec = 0.25

        x_range, _ = self.p_raw.viewRange()
        center = (x_range[0] + x_range[1]) / 2
        self.p_raw.setXRange(
            center - self.view_window_sec / 2,
            center + self.view_window_sec / 2,
            padding=0,
        )

    def render_all(self) -> None:
        vr = self.p_raw.viewRange()
        s_sec, e_sec = vr[0][0], vr[0][1]

        s = int(max(0, s_sec * self.fs))
        e = int(min(self.n_samples, e_sec * self.fs))

        if (e - s) < 50:
            return

        x = np.linspace(s / self.fs, (e - 1) / self.fs, e - s)
        chunk = self.raw[self.current_channel][s:e] * 1e6  # Convert back to microvolts
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
            for ripple in self.ripples[self.current_channel]
            if (ripple.end_sec * self.fs >= s) and (ripple.start_sec * self.fs <= e)
        ]

        for ripple in in_view:
            # Convert seconds to samples and clip to the current view [s, e]
            r_s = int(max(s, ripple.start_sec * self.fs)) - s
            r_e = int(min(e, ripple.end_sec * self.fs)) - s

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

        self.nfft: int = max(1, min(self.nfft, len(chunk)))
        noverlap = int(self.nfft * 0.9)
        noverlap: int = min(noverlap, self.nfft - 1)
        f, t, sxx = spectrogram(
            chunk, fs=self.fs, nperseg=self.nfft, noverlap=noverlap, window='hann'
        )
        mask = (f >= self.spect_low) & (f <= self.spect_high)

        if np.any(mask):
            s_log = 10 * np.log10(sxx[mask, :] + 1e-12)
            s_z = (s_log - np.mean(s_log, axis=1, keepdims=True)) / (
                np.std(s_log, axis=1, keepdims=True) + 1e-6
            )

            self.img.setImage(s_z.T, levels=[self.z_min, self.z_max])

            self.win.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.win.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            self.img.setRect(
                pg.QtCore.QRectF(
                    float(s_sec),
                    float(self.spect_low),
                    float(e_sec - s_sec),
                    float(self.spect_high - self.spect_low),
                )
            )
            self.p_spec.setYRange(self.spect_low, self.spect_high, padding=0)

    def keyPressEvent(self, a0):  # noqa: N802
        if a0.key() == Qt.Key.Key_Right:
            self.go_next_ripple()
        elif a0.key() == Qt.Key.Key_Left:
            self.go_prev_ripple()
        elif a0.key() == Qt.Key.Key_Down:
            self.go_next_channel()
        elif a0.key() == Qt.Key.Key_Up:
            self.go_prev_channel()
        elif a0.key() == Qt.Key.Key_Space:
            self.toggle_ripple_highlight()
        else:
            super().keyPressEvent(a0)

    def show(self) -> None:
        super().show()
        if self.app and getattr(self, '_owns_app', False):
            sys.exit(self.app.exec())

    def showMaximized(self) -> None:  # noqa: N802
        super().showMaximized()
        if self.app and getattr(self, '_owns_app', False):
            sys.exit(self.app.exec())
