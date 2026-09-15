"""Dock panel for dyes: colour a bit (or a substructure) and watch the colour
spread along the dependency pattern of the step mappings."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from .session import Session


class DyePanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("dye selected cell / structure  (D)")
        self.add_btn.setToolTip("Give the selected bit (or the whole active substructure) a colour")
        self.add_btn.clicked.connect(self._add)
        row.addWidget(self.add_btn)
        self.color_btn = QtWidgets.QPushButton("pick colour…")
        self.color_btn.clicked.connect(self._pick_new_color)
        row.addWidget(self.color_btn)
        self.clear_btn = QtWidgets.QPushButton("clear")
        self.clear_btn.clicked.connect(session.clear_dyes)
        row.addWidget(self.clear_btn)
        lay.addLayout(row)
        self.next_color = None
        hint = QtWidgets.QLabel(
            "Every step mixes a bit's dye into every bit it feeds (an output bit gets the average of its "
            "sources): θ spreads it to 11 cells, χ to 3, ρ and π just carry it.  The total amount of each dye "
            "is conserved, so after enough rounds every cell holds the same mixture.  Brightness = concentration "
            "relative to the strongest cell; hue = the mix of dyes.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa;")
        lay.addWidget(hint)
        self.list = QtWidgets.QListWidget()
        self.list.setToolTip("double-click to change a dye's colour; Delete key removes it")
        self.list.itemDoubleClicked.connect(self._recolor)
        lay.addWidget(self.list, 1)
        self.spread = QtWidgets.QLabel("")
        self.spread.setWordWrap(True)
        lay.addWidget(self.spread)
        session.dyesChanged.connect(self.refresh)
        session.positionChanged.connect(lambda *_: self.refresh())
        session.runChanged.connect(self.refresh)
        self.refresh()

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Delete and self.list.currentRow() >= 0:
            self.session.remove_dye(self.list.currentRow())
        else:
            super().keyPressEvent(ev)

    def _add(self) -> None:
        self.session.add_dye(self.session.selected, self.next_color)
        self.next_color = None
        self.color_btn.setText("pick colour…")
        self.color_btn.setStyleSheet("")

    def _pick_new_color(self) -> None:
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor("#ff3b3b"), self, "colour for the next dye")
        if c.isValid():
            self.next_color = c.name()
            self.color_btn.setText(f"next: {c.name()}")
            self.color_btn.setStyleSheet(f"background: {c.name()}; color: black;")

    def _recolor(self, item: QtWidgets.QListWidgetItem) -> None:
        i = self.list.row(item)
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.session.dyes[i].color), self, "dye colour")
        if c.isValid():
            self.session.set_dye_color(i, c.name())

    def refresh(self) -> None:
        s = self.session
        self.list.clear()
        for i, d in enumerate(s.dyes):
            item = QtWidgets.QListWidgetItem(f"dye {i + 1}: {d.label}  ({len(d.origins)} bit(s))  {d.color}")
            item.setForeground(QtGui.QColor(d.color))
            self.list.addItem(item)
        spread = s.dye_spread()
        if spread is None:
            self.spread.setText("No dye yet.  Click a cell (optionally with a substructure chip active) and press D.")
        else:
            k = s.snap_index
            arr = s.dye_array()
            reached = int((arr[min(k, len(arr) - 1)].sum(axis=1) > 0).sum())
            self.spread.setText(f"cells reached: {reached} / 1600.   unevenness max/mean = {spread:.2f}  "
                                f"(1.00 = perfectly homogeneous).")
