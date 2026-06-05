from collections.abc import Callable
from typing import Protocol

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from espresso.ui.state.ripple_viewer_state import RippleViewerState


class ChannelChangeCallback(Protocol):
    def __call__(self, *, channel_name: str) -> None: ...


class TopBar(QWidget):
    """Reusable Top Bar for navigation."""

    def __init__(
        self,
        parent: QWidget,
        on_prev_channel: Callable[[], None],
        on_next_channel: Callable[[], None],
        on_prev_ripple: Callable[[], None],
        on_next_ripple: Callable[[], None],
        on_channel_change: ChannelChangeCallback,
    ):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.prev_ch_btn = QPushButton("Ch -")
        self.next_ch_btn = QPushButton("Ch +")
        self.ch_input = QLineEdit()
        self.ch_input.setFixedWidth(80)
        self.ch_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ch_input.setStyleSheet("""
            QLineEdit { font-weight: bold; font-size: 14px; border: 1px solid #999; border-radius: 4px; padding: 2px; }
        """)

        self.info_label = QLabel("0/0")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.prev_btn = QPushButton("<")
        self.next_btn = QPushButton(">")

        for b in [self.prev_ch_btn, self.next_ch_btn, self.prev_btn, self.next_btn]:
            width = 60 if "Ch" in b.text() else 40
            b.setFixedSize(width, 30)
            b.setStyleSheet(
                "font-weight: bold; border: 1px solid #999; border-radius: 4px;"
            )

        layout.addWidget(self.prev_ch_btn)
        layout.addWidget(self.ch_input)
        layout.addWidget(self.next_ch_btn)
        layout.addStretch()
        layout.addWidget(self.info_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)

        # Store callback functions
        self._on_prev_channel = on_prev_channel
        self._on_next_channel = on_next_channel
        self._on_prev_ripple = on_prev_ripple
        self._on_next_ripple = on_next_ripple
        self._on_channel_change = on_channel_change

        # Connect signals to callbacks
        if self._on_prev_channel:
            self.prev_ch_btn.clicked.connect(self._on_prev_channel)
        if self._on_next_channel:
            self.next_ch_btn.clicked.connect(self._on_next_channel)
        if self._on_prev_ripple:
            self.prev_btn.clicked.connect(self._on_prev_ripple)
        if self._on_next_ripple:
            self.next_btn.clicked.connect(self._on_next_ripple)
        if self._on_channel_change:
            self.ch_input.returnPressed.connect(
                lambda: self._on_channel_change(channel_name=self.ch_input.text())
            )

    def build(self, ripple_viewer_state: RippleViewerState):
        self.ch_input.setText(ripple_viewer_state.channel_name)
        self.info_label.setText(
            f"{ripple_viewer_state.current_ripple_index + 1} / {len(ripple_viewer_state.ripples)}",
        )
