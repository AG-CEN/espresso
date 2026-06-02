from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from espresso.ui.declarative_widgets import Column


class SidebarPanel(QWidget):
    def __init__(self, datasets: dict, callback):
        super().__init__()
        self.datasets = datasets
        self.callback = callback
        self.setFixedWidth(180)
        self.plot_toggles = {}
        self.init_ui()

    def init_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for ds_title in self.datasets.keys():
            lbl = QLabel(ds_title.upper())
            lbl.setStyleSheet(
                "font-weight: bold; margin-top: 8px; color: #444; font-size: 11px;"
            )
            scroll_layout.addWidget(lbl)

            self.plot_toggles[ds_title] = {}
            for trace_type in ["Raw", "Filtered", "Envelope", "Spectrogram"]:
                cb = QCheckBox(f"Show {trace_type}")
                cb.setChecked(True)
                cb.stateChanged.connect(self.callback)
                scroll_layout.addWidget(cb)
                self.plot_toggles[ds_title][trace_type] = cb

        lbl_util = QLabel("UTILITY DOCK")
        lbl_util.setStyleSheet(
            "font-weight: bold; margin-top: 12px; color: #444; font-size: 11px;"
        )
        scroll_layout.addWidget(lbl_util)

        self.dock_cb = QCheckBox("Show Controls")
        self.dock_cb.setChecked(True)
        self.dock_cb.stateChanged.connect(self.callback)
        scroll_layout.addWidget(self.dock_cb)

        scroll.setWidget(scroll_content)
        lay.addWidget(QLabel("PLOT VISIBILITY"))
        lay.addWidget(scroll)

    def link_toggles(self, track_groups: dict):
        for ds_title, tg in track_groups.items():
            tg.toggle_map = self.plot_toggles[ds_title]
