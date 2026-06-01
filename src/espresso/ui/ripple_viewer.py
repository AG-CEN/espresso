import sys

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDial,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from espresso.models.ripple_dataset import RippleDataset
from espresso.ui.ripple_processor import RippleProcessor


class RippleViewer(QWidget):
    def __init__(
        self,
        ripple_datasets: dict[str, RippleDataset],
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
            self.app = QApplication(sys.argv)
            self._owns_app = True
        else:
            self._owns_app = False

        if not ripple_datasets:
            raise ValueError("ripple_dataset list cannot be empty")

        super().__init__()
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(useOpenGL=True)

        self.datasets: dict[str, RippleDataset] = ripple_datasets

        self.primary_ds = list(self.datasets.values())[0]
        self.fs: float = self.primary_ds.fs
        self.current_channel: str = self.primary_ds.get_channels()[0]
        self.n_samples = len(self.primary_ds.raw_volts[self.current_channel])

        self.processor = RippleProcessor(self.fs, spect_low, spect_high)
        self.current_ripple: int = 0
        self.view_window_sec = 2.0

        self.knob_labels = {}
        self.dataset_plots = []

        self.init_ui()

    def init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        sidebar = QWidget()
        sidebar.setFixedWidth(180)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(sidebar)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(right_container)

        top_layout = QHBoxLayout()

        self.prev_ch_btn = QPushButton("Ch -")
        self.next_ch_btn = QPushButton("Ch +")
        self.ch_input = QLineEdit(self.current_channel)
        self.ch_input.setFixedWidth(80)
        self.ch_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ch_input.setStyleSheet("""
            QLineEdit { font-weight: bold; font-size: 14px; border: 1px solid #999; border-radius: 4px; padding: 2px; }
        """)

        self.info_label = QLabel("0/0")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.prev_btn = QPushButton("<")
        self.next_btn = QPushButton(">")

        all_btns = [self.prev_ch_btn, self.next_ch_btn, self.prev_btn, self.next_btn]
        for b in all_btns:
            width = 60 if "Ch" in b.text() else 40
            b.setFixedSize(width, 20)
            b.setStyleSheet(
                "font-weight: bold; border: 1px solid #999; border-radius: 4px;"
            )

        top_layout.addWidget(self.prev_ch_btn)
        top_layout.addWidget(self.ch_input)
        top_layout.addWidget(self.next_ch_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.info_label, alignment=Qt.AlignmentFlag.AlignCenter)
        top_layout.addStretch()
        top_layout.addWidget(self.prev_btn)
        top_layout.addWidget(self.next_btn)

        right_layout.addLayout(top_layout)
        self.ch_input.clearFocus()

        self.win = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.win)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.check_layout = QVBoxLayout(scroll_content)
        self.check_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.plot_toggles = {}
        for ds_title in self.datasets.keys():
            lbl = QLabel(ds_title.upper())
            lbl.setStyleSheet(
                "font-weight: bold; margin-top: 10px; color: #444; font-size: 11px;"
            )
            self.check_layout.addWidget(lbl)

            self.plot_toggles[ds_title] = {}
            for trace_type in ["Raw", "Filtered", "Envelope", "Spectrogram"]:
                cb = QCheckBox(f"Show {trace_type}")
                cb.setChecked(True)
                cb.stateChanged.connect(self._rebuild_plot_grid)
                self.check_layout.addWidget(cb)
                self.plot_toggles[ds_title][trace_type] = cb

        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(QLabel("PLOT VISIBILITY"))
        sidebar_layout.addWidget(scroll)
        self.plot_toggles["Controls"] = {"Dock": QCheckBox("Show Control Dock")}

        lbl_ctrl = QLabel("UTILITY PANELS")
        lbl_ctrl.setStyleSheet(
            "font-weight: bold; margin-top: 15px; color: #444; font-size: 11px;"
        )
        sidebar_layout.addWidget(lbl_ctrl)

        cb_dock = self.plot_toggles["Controls"]["Dock"]
        cb_dock.setChecked(True)
        cb_dock.stateChanged.connect(self._rebuild_plot_grid)
        sidebar_layout.addWidget(cb_dock)

        pg.setConfigOptions(useOpenGL=True, antialias=True)

        self.dataset_plots = []
        self.v_lines = []
        self.master_plot = None

        for idx, (ds_title, ds) in enumerate(self.datasets.items()):
            p_raw = self._add_grilled_plot(f"{ds_title} - Raw LFP")
            p_filt = self._add_grilled_plot(f"{ds_title} - Filtered")
            p_env = self._add_grilled_plot(f"{ds_title} - Envelope")
            p_spec = self._add_grilled_plot(
                f"{ds_title} - Spectrogram", grid_color=(190, 190, 190)
            )

            if idx == 0:
                self.master_plot = p_raw

            p_raw.setXLink(self.master_plot)
            p_filt.setXLink(self.master_plot)
            p_env.setXLink(self.master_plot)
            p_spec.setXLink(self.master_plot)

            c_raw = p_raw.plot(pen=pg.mkPen((33, 33, 33), width=1))
            c_filt = p_filt.plot(pen=pg.mkPen((33, 33, 33), width=1))
            c_env = p_env.plot(pen=pg.mkPen((33, 33, 33), width=1))

            c_raw_hi = p_raw.plot(pen=pg.mkPen("r", width=2.0))
            c_filt_hi = p_filt.plot(pen=pg.mkPen("r", width=2.0))
            c_env_hi = p_env.plot(pen=pg.mkPen("r", width=2.0))

            img = pg.ImageItem()
            img.setLookupTable(pg.colormap.get("turbo").getLookupTable())
            p_spec.addItem(img)

            p_spec.setLabel("bottom", "Time", units="s")

            for p in [p_raw, p_filt, p_env, p_spec]:
                line = pg.InfiniteLine(
                    pos=0,
                    angle=90,
                    pen=pg.mkPen((88, 88, 88), width=2, style=Qt.PenStyle.DashLine),
                )
                p.addItem(line)
                self.v_lines.append(line)

            self.dataset_plots.append(
                {
                    "p_raw": p_raw,
                    "p_filt": p_filt,
                    "p_env": p_env,
                    "p_spec": p_spec,
                    "c_raw": c_raw,
                    "c_filt": c_filt,
                    "c_env": c_env,
                    "c_raw_hi": c_raw_hi,
                    "c_filt_hi": c_filt_hi,
                    "c_env_hi": c_env_hi,
                    "img": img,
                }
            )

        self.knob_layout = QGridLayout()
        self.k_low = self._add_knob("Low Hz", 1, 249, self.processor.spect_low, 0)
        self.k_high = self._add_knob(
            "High Hz", 1, int(self.fs // 2) - 1, self.processor.spect_high, 1
        )
        self.k_nfft = self._add_knob("NFFT", 1, int(self.fs * 0.5), self.processor.nfft, 2)
        self.k_interp = self._add_knob(
            "Z-Interp", 1, 100, int(self.processor.z_interp // 32), col=3, single_step=1
        )

        self.global_colorbar_widget = pg.GraphicsLayoutWidget()
        self.global_colorbar_widget.setFixedSize(60, 70)

        self.global_colorbar = pg.ColorBarItem(
            values=(self.processor.z_min, self.processor.z_max), colorMap="turbo"
        )
        self.global_colorbar_widget.addItem(self.global_colorbar)

        self.knob_layout.addWidget(
            self.global_colorbar_widget,
            1,
            4,
            2,
            1,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self.knob_layout.setColumnStretch(4, 1)

        knob_widget = QWidget()
        knob_widget.setLayout(self.knob_layout)
        knob_widget.setMaximumHeight(95)
        knob_widget.setStyleSheet("background-color: transparent;")

        knob_proxy = QGraphicsProxyWidget()
        knob_proxy.setMinimumHeight(95)
        knob_proxy.setWidget(knob_widget)

        knobs_row = len(self.datasets.keys()) * 4
        self.win.addItem(knob_proxy, row=knobs_row, col=0)
        self.knob_proxy = knob_proxy

        self.p_nav = self.win.addPlot(row=knobs_row + 1, col=0)
        self.p_nav.setMaximumHeight(40)
        self.p_nav.setMouseEnabled(y=False)
        self.p_nav.hideAxis("left")
        self.p_nav.hideAxis("bottom")

        total_duration = self.n_samples / self.fs
        self.p_nav.setLimits(xMin=0, xMax=total_duration, minXRange=total_duration)
        self.p_nav.getViewBox().setBackgroundColor(pg.mkColor(100, 100, 100, 25))

        dec = max(500, self.n_samples // 5000)
        nav_x = np.arange(0, self.n_samples, dec) / self.fs
        self.p_nav.plot(
            nav_x, self.primary_ds.raw_volts[self.current_channel][::dec], pen="k"
        )

        primary_ripples = self.primary_ds.ripples.get(self.current_channel, [])
        if primary_ripples:
            for ripple in primary_ripples:
                start_sec = ripple.start_sec
                end_sec = ripple.end_sec
                width = max(0.001, end_sec - start_sec)
                item = QGraphicsRectItem(start_sec, -1, width, 2)
                item.setPen(pg.mkPen("r", width=0.5))
                item.setBrush(pg.mkBrush(255, 0, 0, 100))
                self.p_nav.addItem(item)

        self.nav_line = pg.InfiniteLine(
            pos=0, movable=False, pen=pg.mkPen("r", width=2)
        )
        self.nav_line.setAcceptHoverEvents(False)
        self.p_nav.addItem(self.nav_line)

        self.prev_btn.clicked.connect(self.go_prev_ripple)
        self.next_btn.clicked.connect(self.go_next_ripple)
        self.prev_ch_btn.clicked.connect(self.go_prev_channel)
        self.next_ch_btn.clicked.connect(self.go_next_channel)
        self.ch_input.returnPressed.connect(self._go_to_channel_from_input)
        self.p_nav.scene().sigMouseClicked.connect(self._on_nav_clicked)

        self.dataset_plots[0]["p_raw"].sigRangeChanged.connect(self._on_plot_moved)
        self.nav_line.sigPositionChanged.connect(self._sync_nav_line_to_view)

        self._rebuild_plot_grid()
        self.go_to_ripple(0)

    def _rebuild_plot_grid(self) -> None:
        self.win.clear()

        current_row = 0
        for ds_title, plots in zip(self.datasets.keys(), self.dataset_plots):
            toggles = self.plot_toggles[ds_title]

            trace_mapping = [
                ("Raw", plots["p_raw"]),
                ("Filtered", plots["p_filt"]),
                ("Envelope", plots["p_env"]),
                ("Spectrogram", plots["p_spec"]),
            ]

            for key, plot_item in trace_mapping:
                if toggles[key].isChecked():
                    self.win.addItem(plot_item, row=current_row, col=0)
                    current_row += 1

        if self.plot_toggles["Controls"]["Dock"].isChecked():
            self.win.addItem(self.knob_proxy, row=current_row, col=0)
            self.win.addItem(self.p_nav, row=current_row + 1, col=0)

        self.win.ci.layout.activate()
        self.render_all()

    def _on_plot_moved(self) -> None:
        view_range = self.dataset_plots[0]["p_raw"].viewRange()[0]
        center_sec = (view_range[0] + view_range[1]) / 2

        self.nav_line.blockSignals(True)
        self.nav_line.setValue(center_sec)
        self.nav_line.blockSignals(False)

        self.render_all()

    def _sync_nav_line_to_view(self) -> None:
        center = self.nav_line.value()
        half_window = self.view_window_sec / 2

        self.dataset_plots[0]["p_raw"].setXRange(
            center - half_window, center + half_window, padding=0
        )

    def _on_nav_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.p_nav.sceneBoundingRect().contains(pos):
                mouse_point = self.p_nav.getViewBox().mapSceneToView(pos)
                new_time = mouse_point.x()
                self.nav_line.setValue(new_time)
                self.render_all()

    def _add_grilled_plot(self, title, grid_color="k"):
        p = pg.PlotItem(title=title)
        p.showGrid(x=True, y=True, alpha=0.3)
        p.setMouseEnabled(y=False)
        grid_pen = pg.mkPen(color=grid_color, width=1)
        p.getAxis("bottom").setPen(grid_pen)
        p.getAxis("left").setPen(grid_pen)
        p.setLimits(
            xMin=0, xMax=self.n_samples / self.fs, maxXRange=self.view_window_sec
        )
        return p

    def _add_knob(self, label, min_v, max_v, cur_v, col, single_step=10):
        k = QDial()
        k.setFixedSize(48, 48)
        k.setRange(min_v, max_v)
        k.setValue(cur_v)
        k.setSingleStep(single_step)
        k.setNotchesVisible(True)
        k.valueChanged.connect(self._on_knob_changed)
        val_lbl = QLabel(f"{label}: {cur_v}")
        val_lbl.setStyleSheet("font-size: 14px; color: #666;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.knob_labels[id(k)] = val_lbl
        self.knob_layout.addWidget(k, 1, col, alignment=Qt.AlignmentFlag.AlignCenter)
        self.knob_layout.addWidget(val_lbl, 2, col)
        self.knob_layout.setColumnStretch(col, 1)

        return k

    def _update_knob_labels(self) -> None:
        self._update_knob_label(self.k_low, self.processor.spect_low)
        self._update_knob_label(self.k_high, self.processor.spect_high)
        self._update_knob_label(self.k_nfft, self.processor.nfft)
        self._update_knob_label(self.k_interp, self.processor.z_interp)

    def _update_knob_label(self, knob, new_value):
        knob_label = self.knob_labels.get(id(knob))
        if knob_label:
            prefix = knob_label.text().split(":")[0]
            knob_label.setText(f"{prefix}: {new_value}")

    def _on_knob_changed(self) -> None:
        self.processor.spect_low = self.k_low.value()
        self.processor.spect_high = self.k_high.value()
        self.processor.nfft = self.k_nfft.value()
        self.processor.z_interp = self.k_interp.value() * 32
        self._update_knob_labels()
        self.render_all()

    def go_to_ripple(self, idx):
        primary_ripples = self.primary_ds.ripples.get(self.current_channel, [])
        if not primary_ripples:
            return
        self.current_ripple = np.clip(idx, 0, len(primary_ripples) - 1)
        ripple = primary_ripples[self.current_ripple]

        for line in self.v_lines:
            line.setPos(ripple.peak_sec)

        self.nav_line.setValue(ripple.peak_sec)
        self.info_label.setText(
            f"RIPPLE {self.current_ripple + 1} / {len(primary_ripples)}"
        )

    def go_next_channel(self) -> None:
        channels = self.primary_ds.get_channels()
        current_idx = channels.index(self.current_channel)
        self._go_to_channel(current_idx + 1)

    def go_prev_channel(self) -> None:
        channels = self.primary_ds.get_channels()
        current_idx = channels.index(self.current_channel)
        self._go_to_channel(current_idx - 1)

    def _go_to_channel(self, idx) -> None:
        channels = self.primary_ds.get_channels()
        self.current_channel = channels[np.clip(idx, 0, len(channels) - 1)]
        self.ch_input.setText(self.current_channel)
        self.go_to_ripple(0)
        self.render_all()

    def _go_to_channel_from_input(self) -> None:
        channel_name = self.ch_input.text()
        self.ch_input.clearFocus()
        if channel_name in self.primary_ds.raw_volts:
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

        x_range, _ = self.dataset_plots[0]["p_raw"].viewRange()
        center = (x_range[0] + x_range[1]) / 2
        self.dataset_plots[0]["p_raw"].setXRange(
            center - self.view_window_sec / 2,
            center + self.view_window_sec / 2,
            padding=0,
        )

    def render_all(self) -> None:
        vr = self.dataset_plots[0]["p_raw"].viewRange()
        s_sec, e_sec = vr[0][0], vr[0][1]

        s = int(max(0, s_sec * self.fs))
        e = int(min(self.n_samples, e_sec * self.fs))

        if (e - s) < 50:
            return

        x = np.linspace(s / self.fs, (e - 1) / self.fs, e - s)

        for idx, (ds, plots) in enumerate(
            zip((self.datasets.values()), self.dataset_plots)
        ):
            if self.current_channel not in ds.raw_volts:
                continue

            chunk = ds.raw_volts[self.current_channel][s:e] * 1e6
            filtered, envelope = self.processor.process_trace(chunk)

            in_view = [
                ripple
                for ripple in ds.ripples.get(self.current_channel, [])
                if (ripple.end_sec * self.fs >= s) and (ripple.start_sec * self.fs <= e)
            ]

            black_mask, hi_raw, hi_filt, hi_env = self.processor.compute_ripple_masks(
                chunk, filtered, envelope, s, e, self.fs, in_view
            )

            clean_raw = chunk.copy().astype(float)
            clean_filt = filtered.copy().astype(float)
            clean_env = envelope.copy().astype(float)

            clean_raw[~black_mask] = np.nan
            clean_filt[~black_mask] = np.nan
            clean_env[~black_mask] = np.nan

            plots["c_raw"].setData(x, clean_raw)
            plots["c_filt"].setData(x, clean_filt)
            plots["c_env"].setData(x, clean_env)

            plots["c_raw_hi"].setData(x, hi_raw)
            plots["c_filt_hi"].setData(x, hi_filt)
            plots["c_env_hi"].setData(x, hi_env)

            f, t, s_z = self.processor.compute_spectrogram(chunk)
            if s_z.size > 0:
                plots["img"].setImage(s_z.T, levels=[self.processor.z_min, self.processor.z_max])
                plots["img"].setRect(self.processor.get_spectrogram_rect(s_sec, e_sec))
                plots["p_spec"].setYRange(self.processor.spect_low, self.processor.spect_high, padding=0)

        self.win.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.win.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    def keyPressEvent(self, a0):
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
        if self.app and getattr(self, "_owns_app", False):
            sys.exit(self.app.exec())

    def showMaximized(self) -> None:  # noqa: N802
        super().showMaximized()
        if self.app and getattr(self, "_owns_app", False):
            sys.exit(self.app.exec())
