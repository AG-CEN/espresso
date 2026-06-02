from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from espresso.models.ripple_dataset import RippleDataset


class LeftPanel(QWidget):
    """Left sidebar panel for controlling plot visibility."""

    # Signal: (dataset_name, plot_type, is_visible)
    plot_visibility_changed = pyqtSignal(str, str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_expanded = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.toggle_btn = QPushButton("▼")
        self.toggle_btn.setMaximumHeight(30)
        self.toggle_btn.setMaximumWidth(30)
        self.toggle_btn.clicked.connect(self.toggle_panel)
        layout.addWidget(self.toggle_btn)

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        self.datasets = {}
        self._items_map = {}  # Map (dataset_name, plot_type) -> QListWidgetItem

    def load_datasets(self, ripple_datasets: dict[str, RippleDataset]) -> None:
        """Load datasets and populate plot checkboxes.

        Args:
            ripple_datasets: Dictionary mapping dataset names to RippleDataset objects.
        """
        self.list_widget.clear()
        self._items_map.clear()
        self.datasets = ripple_datasets

        plot_types = ["raw", "filtered", "hilbert", "spectrogram"]
        plot_labels = {"raw": "Raw", "filtered": "Filtered", "hilbert": "Envelope", "spectrogram": "Spectrogram"}

        for dataset_name in ripple_datasets.keys():
            for plot_type in plot_types:
                label = f"{dataset_name} {plot_labels[plot_type]}"
                
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                
                # Only "raw" is checked by default
                check_state = (
                    Qt.CheckState.Checked
                    if plot_type == "raw"
                    else Qt.CheckState.Unchecked
                )
                item.setCheckState(check_state)
                
                self.list_widget.addItem(item)
                self._items_map[(dataset_name, plot_type)] = item

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Handle checkbox state changes."""
        for (dataset_name, plot_type), map_item in self._items_map.items():
            if map_item is item:
                is_checked = item.checkState() == Qt.CheckState.Checked
                self.plot_visibility_changed.emit(dataset_name, plot_type, is_checked)
                return

    def toggle_panel(self) -> None:
        """Toggle panel expansion."""
        self.is_expanded = not self.is_expanded
        self.list_widget.setVisible(self.is_expanded)
        self.toggle_btn.setText("▼" if self.is_expanded else "▶")
