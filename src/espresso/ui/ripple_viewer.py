import sys

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from espresso.ui.components.signal_plot_renderer import SignalPlotRenderer
from espresso.ui.components.top_bar import TopBar
from espresso.ui.ripple_viewer_controller import RippleViewerController


class RippleViewer(QWidget):
    """Main Orchestrator Frame UI assembly layer."""

    def __init__(self, controller: RippleViewerController):
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

        # 1. Nav Bar Assembly
        self.nav_bar = TopBar(self)
        layout.addWidget(self.nav_bar)

        self.nav_bar.prev_ch_btn.clicked.connect(self.controller.prev_channel)
        self.nav_bar.next_ch_btn.clicked.connect(self.controller.next_channel)
        self.nav_bar.prev_btn.clicked.connect(self.controller.prev_ripple)
        self.nav_bar.next_btn.clicked.connect(self.controller.next_ripple)
        self.nav_bar.ch_input.returnPressed.connect(self._on_channel_input_returned)

        # 2. Main Window Component
        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)

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

        # Link x-axis across plots
        self.p_filt.setXLink(self.p_raw)
        self.p_env.setXLink(self.p_raw)
        self.p_spec.setXLink(self.p_raw)

        # 3. Signal Plot Renderer
        self.plot_renderer = SignalPlotRenderer(
            self.p_raw, self.p_filt, self.p_env, self.p_spec, self.win
        )

        # Add colorbar
        self.win.addItem(self.plot_renderer.get_colorbar(), 3, 1)

        self.nav_bar.ch_input.clearFocus()

        # 6. Plot interactions
        self.p_raw.sigRangeChanged.connect(self._on_plot_range_changed)

        # 7. Set initial view to the first 2s window
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
        """Handle when user pans the raw plot."""
        self.update_ui_from_state()

    def update_ui_from_state(self) -> None:
        c = self.controller

        # Sync simple view states
        ripples_count = len(c.current_ripple_list)
        current_display_idx = c.current_ripple_idx + 1 if ripples_count > 0 else 0
        self.nav_bar.update_display(
            c.current_channel, current_display_idx, ripples_count
        )

        current_ripple = c.current_ripple
        if current_ripple is not None:
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
        """Toggle between 2s and 0.25s zoom levels."""
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
