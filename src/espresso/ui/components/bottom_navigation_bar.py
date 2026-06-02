import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QGraphicsRectItem


class BottomNavigationBar:
    """Manages the full signal miniature breakdown timeline plot component."""

    def __init__(self, win: pg.GraphicsLayoutWidget, row: int):
        self.p_nav = win.addPlot(row=row, col=0)
        self.p_nav.setMaximumHeight(40)
        self.p_nav.setMouseEnabled(y=False)
        self.p_nav.hideAxis("left")
        self.p_nav.hideAxis("bottom")
        self.p_nav.getViewBox().setBackgroundColor(pg.mkColor(100, 100, 100, 25))

        self.nav_line = pg.InfiniteLine(
            pos=0, movable=False, pen=pg.mkPen("r", width=2)
        )

    def update_plot(
        self,
        total_duration: float,
        n_samples: int,
        fs: float,
        raw_signal: np.ndarray,
        ripples: list,
    ):
        self.p_nav.clear()
        self.p_nav.addItem(self.nav_line)
        self.p_nav.setLimits(xMin=0, xMax=total_duration, minXRange=total_duration)

        dec = max(500, n_samples // 5000)
        nav_x = np.arange(0, n_samples, dec) / fs
        self.p_nav.plot(nav_x, raw_signal[::dec], pen="k")

        for ripple in ripples:
            width = max(0.001, ripple.end_sec - ripple.start_sec)
            item = QGraphicsRectItem(ripple.start_sec, -1, width, 2)
            item.setPen(pg.mkPen("r", width=0.5))
            item.setBrush(pg.mkBrush(255, 0, 0, 100))
            self.p_nav.addItem(item)

    def update_line_position(self, pos: float):
        self.nav_line.setPos(pos)
