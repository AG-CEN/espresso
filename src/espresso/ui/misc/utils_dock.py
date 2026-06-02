import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDial,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ControlUtilsDock(QWidget):
    def __init__(
        self, fs: float, n_samples: int, primary_ds, current_channel: str, callback
    ):
        super().__init__()
        self.fs, self.n_samples = fs, n_samples
        self.callback = callback
        self.setMaximumHeight(130)
        self.init_ui(primary_ds, current_channel)

    def init_ui(self, primary_ds, current_channel):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 2, 0, 2)
        main_layout.setSpacing(4)

        bottom_row_layout = QHBoxLayout()
        knob_grid = QGridLayout()

        self.k_low = self._add_knob(knob_grid, "Low Hz", 1, 249, 1, 0)
        self.k_high = self._add_knob(
            knob_grid, "High Hz", 1, int(self.fs // 2) - 1, 250, 1
        )
        self.k_nfft = self._add_knob(
            knob_grid, "NFFT", 1, int(self.fs * 0.5), int(self.fs * 0.125), 2
        )
        self.k_interp = self._add_knob(knob_grid, "Z-Interp", 1, 100, 32, 3)
        bottom_row_layout.addLayout(knob_grid)

        colorbar_win = pg.GraphicsLayoutWidget()
        colorbar_win.setFixedSize(50, 65)
        self.global_colorbar = pg.ColorBarItem(values=(-0.5, 2.0), colorMap="turbo")
        colorbar_win.addItem(self.global_colorbar)
        bottom_row_layout.addWidget(colorbar_win)
        main_layout.addLayout(bottom_row_layout)

        self.p_nav = pg.PlotWidget()
        self.p_nav.setMaximumHeight(35)
        self.p_nav.getViewBox().setMouseEnabled(y=False)
        self.p_nav.getPlotItem().hideAxis("left")
        self.p_nav.getPlotItem().hideAxis("bottom")

        total_duration = self.n_samples / self.fs
        self.p_nav.getPlotItem().setLimits(xMin=0, xMax=total_duration)

        dec = max(500, self.n_samples // 5000)
        nav_x = np.arange(0, self.n_samples, dec) / self.fs
        self.p_nav.plot(nav_x, primary_ds.raw_volts[current_channel][::dec], pen="k")

        self.nav_line = pg.InfiniteLine(
            pos=0, movable=False, pen=pg.mkPen("r", width=2)
        )
        self.p_nav.addItem(self.nav_line)
        main_layout.addWidget(self.p_nav)

    def _add_knob(self, grid, label, min_v, max_v, cur_v, col):
        k = QDial()
        k.setFixedSize(40, 40)
        k.setRange(min_v, max_v)
        k.setValue(cur_v)
        k.valueChanged.connect(self.callback)
        lbl = QLabel(f"{label}: {cur_v}")
        lbl.setStyleSheet("font-size: 10px; color: #555;")
        grid.addWidget(k, 0, col, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(lbl, 1, col, alignment=Qt.AlignmentFlag.AlignCenter)
        return k
