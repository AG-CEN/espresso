import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsProxyWidget, QGridLayout, QWidget


class RipplePlotManager:
    """Manages plot creation, layout, and grid organization."""

    def __init__(self, win: pg.GraphicsLayoutWidget, fs: float, n_samples: int):
        """Initialize plot manager.

        Args:
            win: PyQtGraph GraphicsLayoutWidget.
            fs: Sampling rate in Hz.
            n_samples: Total number of samples.
        """
        self.win = win
        self.fs = fs
        self.n_samples = n_samples
        self.view_window_sec = 2.0

        self.dataset_plots = []
        self.v_lines = []
        self.master_plot = None
        self.knob_proxy = None
        self.p_nav = None

    def add_grilled_plot(self, title: str, grid_color="k") -> pg.PlotItem:
        """Create a plot with grid and configured axes.

        Args:
            title: Plot title.
            grid_color: Grid color (default black).

        Returns:
            Configured PlotItem.
        """
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

    def create_dataset_plots(self, datasets_dict: dict) -> list:
        """Create plot groups for each dataset.

        Args:
            datasets_dict: Dictionary of datasets by name.

        Returns:
            List of plot dictionaries.
        """
        plot_list = []

        for idx, ds_title in enumerate(datasets_dict.keys()):
            p_raw = self.add_grilled_plot(f"{ds_title} - Raw LFP")
            p_filt = self.add_grilled_plot(f"{ds_title} - Filtered")
            p_env = self.add_grilled_plot(f"{ds_title} - Envelope")
            p_spec = self.add_grilled_plot(
                f"{ds_title} - Spectrogram", grid_color=(190, 190, 190)
            )

            if idx == 0:
                self.master_plot = p_raw
            else:
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

            plot_list.append(
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

        self.dataset_plots = plot_list
        return plot_list

    def create_control_panel(
        self, knob_proxy: QGraphicsProxyWidget, n_datasets: int
    ) -> int:
        """Position control panel in grid.

        Args:
            knob_proxy: Graphics proxy widget for controls.
            n_datasets: Number of datasets.

        Returns:
            Row index where controls were placed.
        """
        knobs_row = n_datasets * 4
        self.win.addItem(knob_proxy, row=knobs_row, col=0)
        self.knob_proxy = knob_proxy
        return knobs_row

    def create_navigation_plot(
        self,
        knobs_row: int,
        raw_data: np.ndarray,
        ripples: list,
    ) -> tuple[pg.PlotItem, pg.InfiniteLine]:
        """Create navigation timeline plot.

        Args:
            knobs_row: Row where controls are.
            raw_data: Raw voltage data for preview.
            ripples: List of ripple events.

        Returns:
            Tuple of (plot, infinite_line).
        """
        p_nav = self.win.addPlot(row=knobs_row + 1, col=0)
        p_nav.setMaximumHeight(40)
        p_nav.setMouseEnabled(y=False)
        p_nav.hideAxis("left")
        p_nav.hideAxis("bottom")

        total_duration = self.n_samples / self.fs
        p_nav.setLimits(xMin=0, xMax=total_duration, minXRange=total_duration)
        p_nav.getViewBox().setBackgroundColor(pg.mkColor(100, 100, 100, 25))

        dec = max(500, self.n_samples // 5000)
        nav_x = np.arange(0, self.n_samples, dec) / self.fs
        p_nav.plot(nav_x, raw_data[::dec], pen="k")

        if ripples:
            for ripple in ripples:
                start_sec = ripple.start_sec
                end_sec = ripple.end_sec
                width = max(0.001, end_sec - start_sec)
                from PyQt6.QtWidgets import QGraphicsRectItem

                item = QGraphicsRectItem(start_sec, -1, width, 2)
                item.setPen(pg.mkPen("r", width=0.5))
                item.setBrush(pg.mkBrush(255, 0, 0, 100))
                p_nav.addItem(item)

        nav_line = pg.InfiniteLine(pos=0, movable=False, pen=pg.mkPen("r", width=2))
        nav_line.setAcceptHoverEvents(False)
        p_nav.addItem(nav_line)

        self.p_nav = p_nav
        return p_nav, nav_line

    def rebuild_grid(self, plot_toggles: dict, show_controls: bool) -> None:
        """Rebuild plot grid based on visibility toggles.

        Args:
            plot_toggles: Dictionary of checkbox states per dataset/trace.
            show_controls: Whether to show control dock.
        """
        self.win.clear()

        current_row = 0
        for ds_title, plots in zip(plot_toggles.keys(), self.dataset_plots):
            if ds_title == "Controls":
                continue

            toggles = plot_toggles[ds_title]
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

        if show_controls:
            self.win.addItem(self.knob_proxy, row=current_row, col=0)
            self.win.addItem(self.p_nav, row=current_row + 1, col=0)

        self.win.ci.layout.activate()

    def update_vertical_lines(self, pos: float) -> None:
        """Update vertical reference lines across all plots.

        Args:
            pos: Position in seconds.
        """
        for line in self.v_lines:
            line.setPos(pos)
