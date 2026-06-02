from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


class TopBar(QWidget):
    """Reusable Top Bar for navigation."""

    def __init__(self, parent=None):
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

    def update_display(self, channel: str, current_idx: int, total: int):
        self.ch_input.setText(channel)
        self.info_label.setText(f"{current_idx}/{total}")
