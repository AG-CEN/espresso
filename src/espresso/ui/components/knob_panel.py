from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDial, QGridLayout, QLabel, QWidget


class KnobPanel(QWidget):
    """Reusable control layout representing parameter knobs."""

    def __init__(self, on_changed_callback: Callable[[], None], parent=None):
        super().__init__(parent)
        self.on_changed = on_changed_callback
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("background-color: transparent;")
        self.setMaximumHeight(90)
        self.labels: dict[str, QLabel] = {}

        self.k_low = self._add_knob("Low Hz", 1, 249, 1, 0)
        self.k_high = self._add_knob("High Hz", 1, 250, 250, 1)
        self.k_nfft = self._add_knob("NFFT", 1, 500, 60, 2)
        self.k_interp = self._add_knob("Z-Interp", 1, 100, 32, col=3, single_step=1)

    def _add_knob(
        self,
        label: str,
        min_v: int,
        max_v: int,
        curr_v: int,
        col: int,
        single_step: int = 5,
    ) -> QDial:
        dial = QDial()
        dial.setRange(min_v, max_v)
        dial.setValue(curr_v)
        dial.setSingleStep(single_step)
        dial.valueChanged.connect(self.on_changed)

        lbl = QLabel(f"{label}: {curr_v}")
        self.labels[label] = lbl
        self.layout.addWidget(lbl, 0, col, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(dial, 1, col)
        return dial

    def set_limits(self, high_max: int, nfft_max: int):
        self.k_high.setMaximum(high_max)
        self.k_nfft.setMaximum(nfft_max)

    def update_labels(self, low: int, high: int, nfft: int, interp: int):
        self.labels["Low Hz"].setText(f"Low Hz: {low}")
        self.labels["High Hz"].setText(f"High Hz: {high}")
        self.labels["NFFT"].setText(f"NFFT: {nfft}")
        self.labels["Z-Interp"].setText(f"Z-Interp: {interp}")
