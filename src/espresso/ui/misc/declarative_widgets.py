from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget


class StatelessWidget(QWidget):
    def __init__(self, **kwargs):
        super().__init__()
        self.build(**kwargs)

    def build(self, **kwargs):
        pass


class Column(StatelessWidget):
    def build(self, children: list, spacing: int = 2, margins: tuple = (0, 0, 0, 0)):
        lay = QVBoxLayout(self)
        lay.setSpacing(spacing)
        lay.setContentsMargins(*margins)
        for child in children:
            if isinstance(child, QWidget):
                lay.addWidget(child)
            elif child is None:
                lay.addStretch()


class Row(StatelessWidget):
    def build(self, children: list, spacing: int = 5, margins: tuple = (0, 0, 0, 0)):
        lay = QHBoxLayout(self)
        lay.setSpacing(spacing)
        lay.setContentsMargins(*margins)
        for child in children:
            if isinstance(child, QWidget):
                lay.addWidget(child)
            elif child is None:
                lay.addStretch()


class Padding(StatelessWidget):
    def build(self, child: QWidget, left=0, top=0, right=0, bottom=0):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(left, top, right, bottom)
        lay.addWidget(child)
