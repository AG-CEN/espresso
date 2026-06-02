import sys

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from espresso.models.ripple_dataset import RippleDataset
from espresso.ui.components.left_panel import LeftPanel
from espresso.ui.components.plots_view import PlotsView
from espresso.ui.components.top_bar import TopBar
from espresso.ui.ripple_viewer_controller import RippleViewerController


class RippleViewer(QWidget):
    """Display multiple datasets in columns, each with 4 plots."""

    """Main Orchestrator Frame UI assembly layer.

    Displays multiple datasets simultaneously with shared channel navigation.
    Ripple controls (next/prev) apply only to the first dataset.
    All X-axis ranges are synchronized to the first dataset.
    """

    def __init__(
        self,
        controller: RippleViewerController,
        ripple_datasets: dict[str, RippleDataset] | None = None,
    ):
        # This obtains or creates the QApplication instance used by the viewer.
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
            self._owns_app = True
        else:
            self._owns_app = False

        super().__init__()

        self.controller = controller
        self.ripple_datasets = ripple_datasets or controller.ripple_datasets
        self._last_ripple_idx: int | None = None

        # Store plot renderers and items for each dataset
        self._plot_renderers: dict[str, PlotsView] = {}  # {dataset_name: PlotsView}
        self._plot_items: dict[str, dict[str, pg.PlotItem]] = {}

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(useOpenGL=True, antialias=True)

        self._init_layout()
        self.controller.add_listener(self.update_ui_from_state)
        self.left_panel.plot_visibility_changed.connect(
            self._on_plot_visibility_changed
        )
        self.update_ui_from_state()

    def _init_layout(self) -> None:
        layout = QVBoxLayout(self)

        self.nav_bar = TopBar(self)
        layout.addWidget(self.nav_bar)

        # Connect navigation controls to first dataset only
        self.nav_bar.prev_ch_btn.clicked.connect(self.controller.prev_channel)
        self.nav_bar.next_ch_btn.clicked.connect(self.controller.next_channel)
        self.nav_bar.prev_btn.clicked.connect(self.controller.prev_ripple)
        self.nav_bar.next_btn.clicked.connect(self.controller.next_ripple)
        self.nav_bar.ch_input.returnPressed.connect(self._on_channel_input_returned)

        # This creates the main horizontal layout with left panel and plots.
        main_layout = QHBoxLayout()

        # Left panel for dataset/view selection.
        self.left_panel = LeftPanel(self)
        # Always load from controller's ripple_datasets
        if self.controller.ripple_datasets:
            self.left_panel.load_datasets(self.controller.ripple_datasets)
        main_layout.addWidget(self.left_panel)

        # Right side: scrollable plots area for multiple datasets
        self.win = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.win, stretch=1)

        layout.addLayout(main_layout)

        # Create plot areas for each dataset
        self._create_dataset_plots()

        self.nav_bar.ch_input.clearFocus()

    def _create_dataset_plots(self) -> None:
        """Create plot grid for all datasets (stacked vertically: 4 rows per dataset)."""
        for dataset_idx, dataset_name in enumerate(self.controller.dataset_names):
            dataset = self.controller.get_dataset(dataset_name)
            if dataset is None:
                continue

            # Get dataset duration for axis limits
            first_channel = (
                next(iter(dataset.raw_volts.keys())) if dataset.raw_volts else None
            )
            if first_channel is None:
                continue

            dataset_duration = len(dataset.raw_volts[first_channel]) / dataset.fs

            # Calculate row offset for this dataset (4 plots per dataset)
            row_offset = dataset_idx * 4

            # Create the four plots for this dataset (all in column 0)
            p_raw = self._add_grilled_plot(
                row_offset + 0, 0, f"{dataset_name} Raw LFP", dataset_duration
            )
            p_filt = self._add_grilled_plot(
                row_offset + 1, 0, f"{dataset_name} Filtered", dataset_duration
            )
            p_env = self._add_grilled_plot(
                row_offset + 2, 0, f"{dataset_name} Envelope", dataset_duration
            )
            p_spec = self._add_grilled_plot(
                row_offset + 3,
                0,
                f"{dataset_name} Spectrogram",
                dataset_duration,
                grid_color=(190, 190, 190),
            )

            # Set axis labels
            p_raw.setLabel("left", "Voltage", units="µV")
            p_filt.setLabel("left", "Filtered", units="µV")
            p_env.setLabel("left", "Envelope", units="µV")
            p_spec.setLabel("bottom", "Time", units="s")

            # Link X-axis within this dataset's plots
            p_filt.setXLink(p_raw)
            p_env.setXLink(p_raw)
            p_spec.setXLink(p_raw)

            # Link X-axis to first dataset for synchronized viewing across datasets
            if dataset_idx > 0:
                first_dataset_name = self.controller.dataset_names[0]
                first_raw_plot = self._plot_items[first_dataset_name]["raw"]
                p_raw.setXLink(first_raw_plot)

            # Create renderer for this dataset
            renderer = PlotsView(p_raw, p_filt, p_env, p_spec, self.win, dataset_name)

            # Store references
            self._plot_renderers[dataset_name] = renderer
            self._plot_items[dataset_name] = {
                "raw": p_raw,
                "filtered": p_filt,
                "hilbert": p_env,
                "spectrogram": p_spec,
            }

            # Add colorbar to spectrogram plot's scene (doesn't affect layout grid)
            p_spec.layout.addItem(renderer.colorbar, 2, 3)
            p_spec.layout.setColumnFixedWidth(3, 20)
            # Connect range change signal only for first dataset
            if dataset_idx == 0:
                p_raw.sigRangeChanged.connect(self._on_plot_range_changed)
                first_duration = self.controller.total_duration
                p_raw.setXRange(
                    0,
                    min(self.controller.view_window_sec, first_duration),
                    padding=0,
                )

    def _add_grilled_plot(
        self,
        row: int,
        col: int,
        title: str,
        duration: float,
        grid_color: tuple = ("k",),
    ) -> pg.PlotItem:
        """Create a single plot with grid and axis configuration."""
        p = self.win.addPlot(row=row, col=col, title=title)
        p.showGrid(x=True, y=True, alpha=0.5)
        p.setMouseEnabled(y=False)

        # Configure grid colors
        if grid_color != ("k",):
            grid_pen = pg.mkPen(color=grid_color, width=1)
            p.getAxis("bottom").setPen(grid_pen)
            p.getAxis("left").setPen(grid_pen)
        else:
            grid_pen = pg.mkPen(color="k", width=1)
            p.getAxis("bottom").setPen(grid_pen)
            p.getAxis("left").setPen(grid_pen)

        # Set axis limits based on dataset duration
        p.setLimits(
            xMin=0,
            xMax=duration,
            maxXRange=self.controller.view_window_sec,
        )

        return p

    def _on_channel_input_returned(self) -> None:
        channel_name = self.nav_bar.ch_input.text()
        self.controller.change_channel(channel_name)
        self.update_ui_from_state()

    def _on_plot_range_changed(self) -> None:
        self.update_ui_from_state()

    def _on_plot_visibility_changed(
        self, dataset_name: str, plot_type: str, visible: bool
    ) -> None:
        """Handle plot visibility changes from left panel."""
        # Apply to current channel
        current_channel = self.controller.current_channel
        self.controller.set_plot_visibility(
            dataset_name, current_channel, plot_type, visible
        )
        self.update_ui_from_state()

    def update_ui_from_state(self) -> None:
        """Update all plots based on controller state."""
        c = self.controller

        # Update nav bar with current ripple info (from first dataset)
        ripples_count = len(c.current_ripples) if c.current_ripples else 0
        current_display_idx = c.current_ripple_idx + 1 if ripples_count > 0 else 0
        self.nav_bar.update_display(
            c.current_channel, current_display_idx, ripples_count
        )

        current_ripple = c.current_ripple
        if current_ripple is not None:
            for dataset_name in c.dataset_names:
                self._plot_renderers[dataset_name].update_ripple_marker(
                    current_ripple.peak_sec
                )
            if self._last_ripple_idx != c.current_ripple_idx:
                self._center_view_on_peak(current_ripple.peak_sec)
                self._last_ripple_idx = c.current_ripple_idx
        else:
            self._last_ripple_idx = None

        # Get view range from first dataset's raw plot
        first_dataset_name = c.dataset_names[0]
        if first_dataset_name not in self._plot_items:
            return

        first_raw_plot = self._plot_items[first_dataset_name]["raw"]
        view_range = first_raw_plot.viewRange()[0]
        s_sec, e_sec = view_range[0], view_range[1]

        # Render all visible datasets with the shared channel
        for dataset_name in c.dataset_names:
            dataset = c.get_dataset(dataset_name)
            if dataset is None:
                continue

            # Validate channel exists in this dataset
            if c.current_channel not in dataset.raw_volts:
                continue

            # Get renderer
            if dataset_name not in self._plot_renderers:
                continue

            renderer = self._plot_renderers[dataset_name]
            is_primary = dataset_name == first_dataset_name

            # Control visibility of each plot type
            for plot_type in ["raw", "filtered", "hilbert", "spectrogram"]:
                is_visible = c.get_plot_visibility(
                    dataset_name, c.current_channel, plot_type
                )
                renderer.set_plot_visible(plot_type, is_visible)

            # Get visible plots for this dataset/channel
            visible_plots = c.get_visible_plots(dataset_name, c.current_channel)
            if not visible_plots:
                continue

            renderer.render(
                dataset=dataset,
                channel=c.current_channel,
                s_sec=s_sec,
                e_sec=e_sec,
                sos=c.sos,
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
        first_dataset_name = self.controller.dataset_names[0]
        first_raw_plot = self._plot_items[first_dataset_name]["raw"]
        x_range, _ = first_raw_plot.viewRange()
        center = (x_range[0] + x_range[1]) / 2
        half_window = self.controller.view_window_sec / 2
        first_raw_plot.setXRange(
            max(0, center - half_window),
            min(self.controller.total_duration, center + half_window),
            padding=0,
        )

    def _center_view_on_peak(self, peak_sec: float) -> None:
        # This keeps the selected ripple centered in the raw plot window.
        first_dataset_name = self.controller.dataset_names[0]
        first_raw_plot = self._plot_items[first_dataset_name]["raw"]
        half_window = self.controller.view_window_sec / 2
        first_raw_plot.setXRange(
            max(0, peak_sec - half_window),
            min(self.controller.total_duration, peak_sec + half_window),
            padding=0,
        )

    def run(self) -> None:
        """Start the application event loop."""
        self.showMaximized()
        if self._owns_app and self.app:
            sys.exit(self.app.exec())
