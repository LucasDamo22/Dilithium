"""Shared selection controls: the substructure chips and a cell picker."""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5 import QtCore, QtWidgets

from .session import Session


class StructureLegend(QtWidgets.QWidget):
    """Row of hoverable / checkable chips for row, column, lane, slice, plane, sheet."""

    def __init__(self, session: Session, parent=None, with_pull: bool = True):
        super().__init__(parent)
        from . import anim

        from .flow import FlowLayout

        self.session = session
        lay = FlowLayout(self)
        lay.addWidget(QtWidgets.QLabel("substructures:"))
        self.buttons: Dict[str, QtWidgets.QToolButton] = {}
        tips = {
            "row": "row: 5 bits along x, fixed (y, z) - chi works on rows",
            "column": "column: 5 bits along y, fixed (x, z) - theta sums columns",
            "lane": "lane: 64 bits along z, fixed (x, y) - rho rotates lanes, pi moves them",
            "slice": "slice: 25 bits, fixed z - theta and chi stay inside a slice",
            "plane": "plane: 5 lanes with the same y (320 bits)",
            "sheet": "sheet: 5 lanes with the same x (320 bits) - a theta column parity C[x] is a sheet parity",
        }
        for name in session.STRUCTURES:
            b = QtWidgets.QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setToolTip(tips[name])
            r, g, bl = (int(c * 255) for c in anim.STRUCTURE_COLORS[name])
            b.setStyleSheet(f"QToolButton{{color: rgb({r},{g},{bl}); font-weight: bold;}}"
                            f"QToolButton:checked{{background: rgba({r},{g},{bl},60);}}")
            b.installEventFilter(self)
            b.clicked.connect(lambda checked, n=name: self._clicked(n, checked))
            lay.addWidget(b)
            self.buttons[name] = b
        lay.addSpacing(16)
        b = QtWidgets.QToolButton()
        b.setVisible(with_pull)
        b.setText("pull out  (X)")
        b.setToolTip("Lift the active substructure through the selected cell out of the cube; "
                     "the animation continues with those positions displaced")
        b.setAutoRaise(True)
        b.clicked.connect(lambda: session.pull_out(session.structure, session.selected))
        lay.addWidget(b)
        b = QtWidgets.QToolButton()
        b.setVisible(with_pull)
        b.setText("push back")
        b.setAutoRaise(True)
        b.clicked.connect(lambda: session.push_back())
        lay.addWidget(b)
        lay.addStretch(1)
        self._checked: Optional[str] = None

    def eventFilter(self, obj, ev):
        for name, b in self.buttons.items():
            if obj is b:
                if ev.type() == QtCore.QEvent.Enter:
                    self.session.set_structure(name)
                elif ev.type() == QtCore.QEvent.Leave:
                    self.session.set_structure(self._checked)
        return super().eventFilter(obj, ev)

    def _clicked(self, name: str, checked: bool) -> None:
        self._checked = name if checked else None
        for n, b in self.buttons.items():
            b.setChecked(n == self._checked)
        self.session.set_structure(self._checked)


class CellPickerBar(QtWidgets.QWidget):
    """Pick a cell by coordinates and act on it: track, dye, clear."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        from .flow import FlowLayout

        self.session = session
        lay = FlowLayout(self)
        lay.addWidget(QtWidgets.QLabel("select cell:"))
        self.spins = []
        for name, hi in (("x", 4), ("y", 4), ("z", 63)):
            lay.addWidget(QtWidgets.QLabel(name))
            sp = QtWidgets.QSpinBox()
            sp.setRange(0, hi)
            sp.setToolTip(f"{name} of the cell to select")
            sp.valueChanged.connect(self._apply)
            lay.addWidget(sp)
            self.spins.append(sp)
        for text, tip, slot in (
            ("track  (F)", "Follow this bit (or the whole active substructure) through the permutation",
             lambda: session.track_focus(session.selected)),
            ("dye  (D)", "Dye this bit (or the whole active substructure) from this frame on",
             lambda: session.add_dye(session.selected)),
            ("clear  (Esc)", "Clear the selection", lambda: session.select(None)),
        ):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            lay.addWidget(b)
        self._updating = False
        session.selectionChanged.connect(self._sync)
        self._sync(session.selected)

    def _apply(self, _v) -> None:
        if not self._updating:
            self.session.select(tuple(sp.value() for sp in self.spins))

    def _sync(self, cell) -> None:
        if cell is None:
            return
        self._updating = True
        try:
            for sp, v in zip(self.spins, cell):
                sp.setValue(int(v))
        finally:
            self._updating = False
