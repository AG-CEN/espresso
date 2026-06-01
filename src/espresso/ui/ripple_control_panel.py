import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDial,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class RippleControlPanel:
    """Manages control UI: knobs, buttons, checkboxes."""

    def __init__(self):
        """Initialize control panel."""
        self.knob_labels = {}
        self.plot_toggles = {}
        self.buttons = {}

    def create_channel_controls(
        self, current_channel: str
    ) -> tuple[QPushButton, QPushButton, QLineEdit, QLabel]:
        """Create channel navigation controls.

        Args:
            current_channel: Current channel name.

        Returns:
            Tuple of (prev_btn, next_btn, input, label).
        """
        prev_ch_btn = QPushButton("Ch -")
        next_ch_btn = QPushButton("Ch +")
        ch_input = QLineEdit(current_channel)
        ch_input.setFixedWidth(80)
        ch_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ch_input.setStyleSheet("""
            QLineEdit { font-weight: bold; font-size: 14px; border: 1px solid #999; border-radius: 4px; padding: 2px; }
        """)

        info_label = QLabel("0/0")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        for b in [prev_ch_btn, next_ch_btn]:
            b.setFixedSize(60, 20)
            b.setStyleSheet(
                "font-weight: bold; border: 1px solid #999; border-radius: 4px;"
            )

        self.buttons["prev_ch"] = prev_ch_btn
        self.buttons["next_ch"] = next_ch_btn

        return prev_ch_btn, next_ch_btn, ch_input, info_label

    def create_ripple_nav_buttons(self) -> tuple[QPushButton, QPushButton]:
        """Create ripple navigation buttons.

        Returns:
            Tuple of (prev_btn, next_btn).
        """
        prev_btn = QPushButton("<")
        next_btn = QPushButton(">")

        for b in [prev_btn, next_btn]:
            b.setFixedSize(40, 20)
            b.setStyleSheet(
                "font-weight: bold; border: 1px solid #999; border-radius: 4px;"
            )

        self.buttons["prev_ripple"] = prev_btn
        self.buttons["next_ripple"] = next_btn

        return prev_btn, next_btn

    def create_plot_visibility_panel(self, datasets: dict) -> QWidget:
        """Create sidebar panel for plot visibility toggles.

        Args:
            datasets: Dictionary of datasets by name.

        Returns:
            Scroll widget with checkboxes.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        check_layout = QVBoxLayout(scroll_content)
        check_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.plot_toggles = {}
        for ds_title in datasets.keys():
            lbl = QLabel(ds_title.upper())
            lbl.setStyleSheet(
                "font-weight: bold; margin-top: 10px; color: #444; font-size: 11px;"
            )
            check_layout.addWidget(lbl)

            self.plot_toggles[ds_title] = {}
            for trace_type in ["Raw", "Filtered", "Envelope", "Spectrogram"]:
                cb = QCheckBox(f"Show {trace_type}")
                cb.setChecked(True)
                check_layout.addWidget(cb)
                self.plot_toggles[ds_title][trace_type] = cb

        scroll.setWidget(scroll_content)
        return scroll

    def create_control_dock_checkbox(self) -> QCheckBox:
        """Create checkbox to toggle control dock visibility.

        Returns:
            Checkbox widget.
        """
        cb = QCheckBox("Show Control Dock")
        cb.setChecked(True)
        self.plot_toggles["Controls"] = {"Dock": cb}
        return cb

    def create_knob_panel(
        self,
        min_low: int,
        max_high: int,
        fs: float,
    ) -> tuple[QWidget, dict]:
        """Create knob control panel.

        Args:
            min_low: Minimum low frequency.
            max_high: Maximum high frequency.
            fs: Sampling rate.

        Returns:
            Tuple of (widget, knob_dict).
        """
        knob_layout = QGridLayout()
        knobs = {}

        knobs["low"] = self._add_knob(knob_layout, "Low Hz", 1, 249, 1, 0)
        knobs["high"] = self._add_knob(
            knob_layout, "High Hz", 1, int(fs // 2) - 1, 250, 1
        )
        knobs["nfft"] = self._add_knob(
            knob_layout, "NFFT", 1, int(fs * 0.5), int(fs * 0.125), 2
        )
        knobs["interp"] = self._add_knob(
            knob_layout, "Z-Interp", 1, 100, 32, col=3, single_step=1
        )

        # Add colorbar placeholder
        global_colorbar_widget = pg.GraphicsLayoutWidget()
        global_colorbar_widget.setFixedSize(60, 70)
        global_colorbar = pg.ColorBarItem(values=(-0.5, 2.0), colorMap="turbo")
        global_colorbar_widget.addItem(global_colorbar)

        knob_layout.addWidget(
            global_colorbar_widget,
            1,
            4,
            2,
            1,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        knob_layout.setColumnStretch(4, 1)

        knob_widget = QWidget()
        knob_widget.setLayout(knob_layout)
        knob_widget.setMaximumHeight(95)
        knob_widget.setStyleSheet("background-color: transparent;")

        return knob_widget, knobs

    def _add_knob(
        self,
        layout: QGridLayout,
        label: str,
        min_v: int,
        max_v: int,
        cur_v: int,
        col: int,
        single_step: int = 10,
    ) -> QDial:
        """Add a knob to the layout.

        Args:
            layout: Grid layout to add to.
            label: Knob label.
            min_v: Minimum value.
            max_v: Maximum value.
            cur_v: Current value.
            col: Column index.
            single_step: Step size.

        Returns:
            Configured QDial.
        """
        k = QDial()
        k.setFixedSize(48, 48)
        k.setRange(min_v, max_v)
        k.setValue(cur_v)
        k.setSingleStep(single_step)
        k.setNotchesVisible(True)

        val_lbl = QLabel(f"{label}: {cur_v}")
        val_lbl.setStyleSheet("font-size: 14px; color: #666;")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.knob_labels[id(k)] = val_lbl
        layout.addWidget(k, 1, col, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(val_lbl, 2, col)
        layout.setColumnStretch(col, 1)

        return k

    def update_knob_label(self, knob: QDial, new_value: int) -> None:
        """Update knob value display.

        Args:
            knob: QDial widget.
            new_value: New value to display.
        """
        knob_label = self.knob_labels.get(id(knob))
        if knob_label:
            prefix = knob_label.text().split(":")[0]
            knob_label.setText(f"{prefix}: {new_value}")
