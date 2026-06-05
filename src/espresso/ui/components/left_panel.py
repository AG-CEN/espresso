from typing import Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from espresso.models.ripple_dataset import RippleDataset
from espresso.ui.state.ripple_viewer_state import PlotId, PlotType


class PlotVisibilityCallback(Protocol):
    def __call__(
        self,
        *,
        plot_id: PlotId,
        new_value: bool,
    ) -> None: ...


class LeftPanel(QWidget):
    """Left sidebar panel for controlling plot visibility."""

    def __init__(
        self,
        parent: QWidget,
        ripple_datasets: list[RippleDataset],
        on_plot_visibility_toggled: PlotVisibilityCallback,
    ):
        super().__init__(parent)

        self.ripple_datasets = ripple_datasets
        self.on_plot_visibility_toggled = on_plot_visibility_toggled

        self.is_expanded = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setMaximumHeight(30)
        self.toggle_btn.setMaximumWidth(30)
        self.toggle_btn.clicked.connect(self._toggle_panel)
        layout.addWidget(self.toggle_btn)

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        self._items_map: dict[PlotId, QListWidgetItem] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        plot_types: list[PlotType] = [
            PlotType.raw,
            PlotType.filtered,
            PlotType.envelope,
            PlotType.spectrogram,
        ]

        for ripple_dataset in self.ripple_datasets:
            for plot_type in plot_types:
                item = QListWidgetItem(
                    f"{ripple_dataset.label} {plot_type.name.capitalize()}"
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.list_widget.addItem(item)
                self._items_map[(ripple_dataset.label, plot_type)] = item

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Handle checkbox state changes."""
        for plot_id, map_item in self._items_map.items():
            if map_item is item:
                is_checked: bool = item.checkState() == Qt.CheckState.Checked
                self.on_plot_visibility_toggled(
                    plot_id=plot_id,
                    new_value=is_checked,
                )
                return

    def _toggle_panel(self) -> None:
        """Toggle panel expansion."""
        self.is_expanded = not self.is_expanded
        self.list_widget.setVisible(self.is_expanded)
        self.toggle_btn.setText("▼" if self.is_expanded else "▶")
