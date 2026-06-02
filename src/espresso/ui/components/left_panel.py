from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from espresso.models.ripple_dataset import RippleDataset


class LeftPanel(QWidget):
    """Left sidebar panel displaying plot options."""

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

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Plots")
        self.tree.setColumnCount(1)
        layout.addWidget(self.tree)

        self.datasets = {}

    def load_datasets(self, ripple_datasets: dict[str, RippleDataset]) -> None:
        """Load datasets and populate plot options.

        Args:
            ripple_datasets: Dictionary mapping dataset names to RippleDataset objects.
        """
        self.tree.clear()
        self.datasets = ripple_datasets

        plot_types = ["raw", "filtered", "envelope", "spectrogram"]

        for dataset_name in ripple_datasets.keys():
            for plot_type in plot_types:
                item = QTreeWidgetItem()
                item.setText(0, f"{dataset_name} - {plot_type}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked)
                self.tree.addTopLevelItem(item)

    def toggle_panel(self) -> None:
        """Toggle panel expansion."""
        self.is_expanded = not self.is_expanded
        self.tree.setVisible(self.is_expanded)
        self.toggle_btn.setText("▼" if self.is_expanded else "▶")
