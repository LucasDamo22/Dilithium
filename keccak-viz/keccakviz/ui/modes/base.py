"""Shared bits for the QPainter-based 2D modes."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import keccak as K
from .. import anim
from ..session import Session


def qcolor(rgb, alpha: float = 1.0) -> QtGui.QColor:
    r, g, b = (int(max(0.0, min(1.0, c)) * 255) for c in rgb[:3])
    return QtGui.QColor(r, g, b, int(alpha * 255))


C_ONE = qcolor(anim.COL_ONE)
C_ZERO = qcolor(anim.COL_ZERO)
C_TO_ONE = qcolor(anim.COL_CHANGED_TO_ONE)
C_TO_ZERO = qcolor(anim.COL_CHANGED_TO_ZERO)
C_UNCH_ONE = qcolor(anim.COL_UNCHANGED_ONE)
C_DIFF = qcolor(anim.COL_DIFF)
C_DIFF_DIM = qcolor(anim.COL_DIFF_DIM_ONE)
C_BG = QtGui.QColor(23, 25, 32)
C_PANEL = QtGui.QColor(37, 39, 46)
C_TEXT = QtGui.QColor(225, 225, 230)
C_DIM = QtGui.QColor(150, 150, 160)
C_SEL = QtGui.QColor(255, 255, 255)
C_SRC = QtGui.QColor(50, 255, 230)
C_RATE = QtGui.QColor(60, 120, 90)
C_CAP = QtGui.QColor(120, 60, 90)
C_PAD = QtGui.QColor(200, 120, 40)
MONO = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)


def bit_colors(session: Session, snap: Optional[K.Snapshot] = None) -> np.ndarray:
    """(5,5,64) array of QColor-able RGB triples following the session colour mode."""
    snap = snap or session.snapshot
    cur = K.lanes_to_bits(snap.state).reshape(-1)
    prev_snap = session.trace[max(0, snap.index - 1)]
    prev = K.lanes_to_bits(prev_snap.state).reshape(-1)
    diff = None
    if session.color_mode == "avalanche":
        diff = session.avalanche().diff_bits(snap.index).reshape(-1)
    col, _ = anim.static_colors(cur, prev, session.color_mode, diff)
    return col.reshape(5, 5, 64, 3)


class ModeWidget(QtWidgets.QWidget):
    """Base: repaints on every session change; subclasses implement ``paint``."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._hover: Optional[Tuple[int, int, int]] = None
        session.runChanged.connect(self.on_run_changed)
        session.positionChanged.connect(lambda *_: self.update())
        session.selectionChanged.connect(lambda *_: self.update())
        session.structureChanged.connect(lambda *_: self.update())
        session.colorModeChanged.connect(lambda *_: self.update())
        session.trackedChanged.connect(self.update)

    def on_run_changed(self) -> None:
        self.update()

    def activated(self) -> None:
        """Called when the mode becomes visible."""
        self.update()

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        p.fillRect(self.rect(), C_BG)
        try:
            self.paint(p)
        finally:
            p.end()

    def paint(self, p: QtGui.QPainter) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def cell_at(self, px: int, py: int) -> Optional[Tuple[int, int, int]]:
        return None

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        c = self.cell_at(ev.x(), ev.y())
        if c != self._hover:
            self._hover = c
            self.update()

    def leaveEvent(self, ev) -> None:
        self._hover = None
        self.update()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            c = self.cell_at(ev.x(), ev.y())
            self.session.select(c)

    def draw_header(self, p: QtGui.QPainter, text: str, sub: str = "") -> int:
        f = p.font()
        f.setPointSize(12)
        f.setBold(True)
        p.setFont(f)
        p.setPen(C_TEXT)
        p.drawText(12, 22, text)
        f.setPointSize(9)
        f.setBold(False)
        p.setFont(f)
        if sub:
            p.setPen(C_DIM)
            p.drawText(12, 40, sub)
            return 50
        return 32

    def draw_tracked(self, p: QtGui.QPainter, rect_of, label: bool = True) -> None:
        """Outline every tracked bit; ``rect_of(x, y, z)`` gives its rectangle."""
        s = self.session
        for ci, cell in s.tracked_big():
            r = rect_of(*cell)
            if r is not None:
                col = QtGui.QColor(s.TRACK_COLORS[ci])
                col.setAlpha(110)
                p.fillRect(r, col)
        for ci, cell, _t in s.tracked_small():
            r = rect_of(*cell)
            if r is None:
                continue
            col = QtGui.QColor(s.TRACK_COLORS[ci])
            p.setPen(QtGui.QPen(col, 2))
            p.drawRect(r)
            if label:
                p.setPen(col)
                p.drawText(QtCore.QPointF(r.right() + 2, r.top() + 8), f"#{ci + 1}")

    def draw_cell_info(self, p: QtGui.QPainter, cell, y: int) -> None:
        s = self.session
        snap = s.snapshot
        x, yy, z = cell
        bits = K.lanes_to_bits(snap.state)
        txt = (f"cell (x={x}, y={yy}, z={z})  lane {K.lane_index(x, yy)} = A[{x},{yy}]  "
               f"bit {K.bit_index(x, yy, z)}  value {int(bits[x, yy, z])}  rho offset {int(K.RHO_OFFSETS[x, yy])}")
        p.setPen(C_TEXT)
        p.drawText(12, y, txt)
