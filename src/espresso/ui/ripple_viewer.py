import sys

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from espresso.ui.components.left_panel import LeftPanel
from espresso.ui.components.plots_view import PlotsView
from espresso.ui.components.top_bar import TopBar
from espresso.ui.state.ripple_viewer_controller import RippleViewerController
from espresso.ui.state.ripple_viewer_state import RippleViewerState


class RippleViewer(QWidget):
    """Display multiple datasets in columns, each with 4 plots."""

    def __init__(
        self,
        controller: RippleViewerController,
    ):
        self.app = QApplication(sys.argv)
        self._owns_app = True

        super().__init__()

        self.controller = controller
        self.ripple_datasets = controller.ripple_datasets
        self._last_ripple_idx: int | None = None

        self._init_ui()

        self._state_subscription = self.controller._state_subject.subscribe(self.build)

    def _init_ui(self) -> None:
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(useOpenGL=True, antialias=True)

        layout = QVBoxLayout(self)

        self.top_bar = TopBar(
            self,
            on_prev_channel=self.controller.prev_channel,
            on_next_channel=self.controller.next_channel,
            on_prev_ripple=self.controller.prev_ripple,
            on_next_ripple=self.controller.next_ripple,
            on_channel_change=self.controller.change_channel,
        )

        layout.addWidget(self.top_bar)

        main_layout = QHBoxLayout()

        self.left_panel = LeftPanel(
            self,
            ripple_datasets=self.controller.ripple_datasets,
            on_plot_visibility_toggled=self.controller.on_plot_visibility_toggled,
        )
        main_layout.addWidget(self.left_panel)

        right_plots_layout = QVBoxLayout()

        self.plots_views = [
            PlotsView(
                ripple_dataset=ripple_dataset,
                initial_state=self.controller.state,
            )
            for ripple_dataset in self.ripple_datasets
        ]

        reference_plot = self.plots_views[0].p_raw
        for plots_view in self.plots_views:
            right_plots_layout.addWidget(plots_view)
            plots_view.set_x_link(reference_plot=reference_plot)

        main_layout.addLayout(right_plots_layout, stretch=4)

        layout.addLayout(main_layout)

        # self.top_bar.ch_input.clearFocus()

    def build(self, ripple_viewer_state: RippleViewerState) -> None:
        """Update UI based on state."""
        self.top_bar.build(ripple_viewer_state=ripple_viewer_state)
        for plots_view in self.plots_views:
            plots_view.build(ripple_viewer_state=ripple_viewer_state)

    def keyPressEvent(self, a0) -> None:
        if a0.key() == Qt.Key.Key_Right:
            self.controller.next_ripple()
        elif a0.key() == Qt.Key.Key_Left:
            self.controller.prev_ripple()
        elif a0.key() == Qt.Key.Key_Down:
            self.controller.next_channel()
        elif a0.key() == Qt.Key.Key_Up:
            self.controller.prev_channel()
        elif a0.key() == Qt.Key.Key_Space:
            self.controller.toggle_ripple_highlight()
        else:
            super().keyPressEvent(a0)

    def run(self) -> None:
        """Start the application event loop."""
        self.showMaximized()
        if self._owns_app and self.app:
            sys.exit(self.app.exec())
