import sys

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from espresso.ui.components.plots_view import PlotsView
from espresso.ui.components.top_bar import TopBar
from espresso.ui.ripple_viewer_controller import RippleViewerController


class RippleViewer(QWidget):
    """Main Orchestrator Frame UI assembly layer."""

    def __init__(self, controller: RippleViewerController):
        # This obtains or creates the QApplication instance used by the viewer.
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
            self._owns_app = True
        else:
            self._owns_app = False

        super().__init__()

        self.controller = controller
        self._last_ripple_idx: int | None = None

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(useOpenGL=True, antialias=True)

        self._init_layout()
        self.controller.add_listener(self.update_ui_from_state)
        self.update_ui_from_state()

    def _init_layout(self) -> None:
        layout = QVBoxLayout(self)

        self.nav_bar = TopBar(self)
        layout.addWidget(self.nav_bar)

        self.nav_bar.prev_ch_btn.clicked.connect(self.controller.prev_channel)
        self.nav_bar.next_ch_btn.clicked.connect(self.controller.next_channel)
        self.nav_bar.prev_btn.clicked.connect(self.controller.prev_ripple)
        self.nav_bar.next_btn.clicked.connect(self.controller.next_ripple)
        self.nav_bar.ch_input.returnPressed.connect(self._on_channel_input_returned)

        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

        # This creates the four stacked plots for raw, filtered, envelope, and spectrogram.
        self.p_raw = self._add_grilled_plot(0, "Raw LFP")
        self.p_filt = self._add_grilled_plot(1, "Filtered")
        self.p_env = self._add_grilled_plot(2, "Envelope")
        self.p_spec = self._add_grilled_plot(
            3, "Spectrogram", grid_color=(190, 190, 190)
        )

        self.p_raw.setLabel("left", "Voltage", units="µV")
        self.p_filt.setLabel("left", "Filtered", units="µV")
        self.p_env.setLabel("left", "Envelope", units="µV")
        self.p_spec.setLabel("bottom", "Time", units="s")

        # This keeps horizontal zoom and pan synchronized across the three signal plots.
        self.p_filt.setXLink(self.p_raw)
        self.p_env.setXLink(self.p_raw)
        self.p_spec.setXLink(self.p_raw)

        self.plot_renderer = PlotsView(
            self.p_raw, self.p_filt, self.p_env, self.p_spec, self.win
        )

        self.win.addItem(self.plot_renderer.colorbar, 3, 1)

        self.nav_bar.ch_input.clearFocus()

        # This updates plot contents when the raw plot range changes.
        self.p_raw.sigRangeChanged.connect(self._on_plot_range_changed)

        self.p_raw.setXRange(
            0,
            min(self.controller.view_window_sec, self.controller.total_duration),
            padding=0,
        )

    def _add_grilled_plot(
        self, row: int, title: str, grid_color: tuple = ("k",)
    ) -> pg.PlotItem:
        p = self.win.addPlot(row=row, col=0, title=title)
        p.showGrid(x=True, y=True, alpha=0.5)
        p.setMouseEnabled(y=False)

        if grid_color != ("k",):
            grid_pen = pg.mkPen(color=grid_color, width=1)
            p.getAxis("bottom").setPen(grid_pen)
            p.getAxis("left").setPen(grid_pen)
        else:
            grid_pen = pg.mkPen(color="k", width=1)
            p.getAxis("bottom").setPen(grid_pen)
            p.getAxis("left").setPen(grid_pen)

        p.setLimits(
            xMin=0,
            xMax=self.controller.total_duration,
            maxXRange=self.controller.view_window_sec,
        )

        return p

    def _on_channel_input_returned(self) -> None:
        channel_name = self.nav_bar.ch_input.text()
        self.controller.change_channel(channel_name)
        if channel_name not in self.controller.channels:
            self.update_ui_from_state()

    def _on_plot_range_changed(self) -> None:
        self.update_ui_from_state()

    def update_ui_from_state(self) -> None:
        c = self.controller

        ripples_count = len(c.current_ripple_list)
        current_display_idx = c.current_ripple_idx + 1 if ripples_count > 0 else 0
        self.nav_bar.update_display(
            c.current_channel, current_display_idx, ripples_count
        )

        current_ripple = c.current_ripple
        if current_ripple is not None:
            # This updates the peak marker and recenters the raw plot when a new ripple is selected.
            self.plot_renderer.update_ripple_marker(current_ripple.peak_sec)
            if self._last_ripple_idx != c.current_ripple_idx:
                self._center_view_on_peak(current_ripple.peak_sec)
                self._last_ripple_idx = c.current_ripple_idx
        else:
            self._last_ripple_idx = None

        view_range = self.p_raw.viewRange()[0]
        s_sec, e_sec = view_range[0], view_range[1]

        self.plot_renderer.render(
            raw_signal=c.raw[c.current_channel],
            fs=c.fs,
            sos=c.sos,
            ripples=c.current_ripple_list,
            s_sec=s_sec,
            e_sec=e_sec,
            spect_low=c.spect_low,
            spect_high=c.spect_high,
            nfft=c.nfft,
            z_min=c.z_min,
            z_max=c.z_max,
        )

    def keyPressEvent(self, a0) -> None:  # noqa: N802
        if a0.key() == Qt.Key.Key_Right:
            self.controller.next_ripple()
        elif a0.key() == Qt.Key.Key_Left:
            self.controller.prev_ripple()
        elif a0.key() == Qt.Key.Key_Down:
            self.controller.next_channel()
        elif a0.key() == Qt.Key.Key_Up:
            self.controller.prev_channel()
        elif a0.key() == Qt.Key.Key_Space:
            self._toggle_zoom()
        else:
            super().keyPressEvent(a0)

    def _toggle_zoom(self) -> None:
        # This toggles the view window size while keeping the current center time.
        self.controller.toggle_ripple_highlight()
        x_range, _ = self.p_raw.viewRange()
        center = (x_range[0] + x_range[1]) / 2
        half_window = self.controller.view_window_sec / 2
        self.p_raw.setXRange(
            max(0, center - half_window),
            min(self.controller.total_duration, center + half_window),
            padding=0,
        )

    def _center_view_on_peak(self, peak_sec: float) -> None:
        # This keeps the selected ripple centered in the raw plot window.
        half_window = self.controller.view_window_sec / 2
        self.p_raw.setXRange(
            max(0, peak_sec - half_window),
            min(self.controller.total_duration, peak_sec + half_window),
            padding=0,
        )

    def run(self) -> None:
        """Starts the desktop engine event loop if this instance spawned it."""
        self.showMaximized()
        if self._owns_app and self.app:
            sys.exit(self.app.exec())
