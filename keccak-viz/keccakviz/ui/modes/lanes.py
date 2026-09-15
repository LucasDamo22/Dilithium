"""Lane table: 25 lanes as 64-bit hex values in a 5x5 grid, x across, y down."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import keccak as K
from .base import C_DIM, C_SEL, C_TEXT, C_TO_ONE, C_UNCH_ONE, MONO, ModeWidget


class LaneTable(ModeWidget):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.setFont(MONO)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(12, 56, 12, 0)
        bar.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.copy_btn = QtWidgets.QPushButton("copy table as text")
        self.copy_btn.clicked.connect(self.copy_text)
        bar.addWidget(self.copy_btn)
        self.show_xor = QtWidgets.QCheckBox("show XOR with previous step")
        self.show_xor.setChecked(True)
        self.show_xor.toggled.connect(lambda _: self.update())
        bar.addWidget(self.show_xor)
        bar.addStretch(1)
        self.setLayout(bar)

    def text_dump(self) -> str:
        s = self.session
        snap = s.snapshot
        lines = [f"# {s.position_text()}", "#        x=0              x=1              x=2              x=3              x=4"]
        for y in range(5):
            lines.append(f"y={y}  " + "  ".join(f"{int(snap.state[x, y]):016x}" for x in range(5)))
        if snap.theta_c is not None:
            lines.append("C    " + "  ".join(f"{int(v):016x}" for v in snap.theta_c))
            lines.append("D    " + "  ".join(f"{int(v):016x}" for v in snap.theta_d))
        if snap.round_constant is not None:
            lines.append(f"RC[{snap.round_index_abs}] = {snap.round_constant:016x}")
        return "\n".join(lines)

    def copy_text(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.text_dump())

    def _geom(self):
        w = self.width()
        colw = max(120, min(190, (w - 130) // 5))
        return 110, 120, colw, 52

    def cell_at(self, px, py):
        top, x0, colw, rowh = self._geom()
        x = (px - x0) // colw
        y = (py - top) // rowh
        if 0 <= x < 5 and 0 <= y < 5:
            return int(x), int(y), 0
        return None

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        snap = s.snapshot
        prev = s.prev_snapshot
        self.draw_header(p, f"Lane table — {s.position_text()}",
                         "A[x,y] as 64-bit little-endian words, x across, y down; lane index = x + 5y, bytes 8·(x+5y)… of the state.")
        top, x0, colw, rowh = self._geom()
        f = QtGui.QFont(MONO)
        f.setPointSize(max(7, min(11, colw // 15)))
        p.setFont(f)
        p.setPen(C_DIM)
        for x in range(5):
            p.drawText(x0 + x * colw, top - 6, f"x={x}")
        sel = s.selected
        for y in range(5):
            p.setPen(C_DIM)
            p.drawText(12, top + y * rowh + 18, f"y={y}")
            p.drawText(12, top + y * rowh + 34, f"lanes {5 * y}–{5 * y + 4}")
            for x in range(5):
                v = int(snap.state[x, y])
                pv = int(prev.state[x, y])
                r = QtCore.QRect(x0 + x * colw, top + y * rowh, colw - 6, rowh - 6)
                changed = v != pv and snap.step != "initial"
                p.fillRect(r, QtGui.QColor(50, 45, 40) if changed else QtGui.QColor(34, 36, 44))
                if sel and (sel[0], sel[1]) == (x, y):
                    p.setPen(QtGui.QPen(C_SEL, 2))
                    p.drawRect(r)
                if s.structure in ("lane", "row", "column", "plane", "sheet", "slice") and (sel or self._hover):
                    ax, ay, _ = sel or self._hover
                    hit = {"lane": (x, y) == (ax, ay), "row": y == ay, "column": x == ax, "plane": y == ay,
                           "sheet": x == ax, "slice": True}[s.structure]
                    if hit:
                        p.setPen(QtGui.QPen(QtGui.QColor(255, 220, 80), 1))
                        p.drawRect(r.adjusted(1, 1, -1, -1))
                p.setPen(C_TO_ONE if changed else C_TEXT)
                p.drawText(r.left() + 6, r.top() + 18, f"{v:016x}")
                if self.show_xor.isChecked() and snap.step != "initial":
                    p.setPen(C_DIM)
                    p.drawText(r.left() + 6, r.top() + 36, f"^{v ^ pv:016x}" if changed else " (unchanged)")
                if not self.show_xor.isChecked() or snap.step == "initial":
                    p.setPen(C_DIM)
                    p.drawText(r.left() + 6, r.top() + 36, f"lane {x + 5 * y}  r={int(K.RHO_OFFSETS[x, y])}")
        y_extra = top + 5 * rowh + 16
        if snap.theta_c is not None:
            p.setPen(QtGui.QColor(90, 220, 240))
            p.drawText(12, y_extra, "C[x] = parity of sheet x:   " + "  ".join(f"{int(v):016x}" for v in snap.theta_c))
            p.setPen(QtGui.QColor(170, 140, 255))
            p.drawText(12, y_extra + 20, "D[x] = C[x-1] ^ ROT(C[x+1],1): " + "  ".join(f"{int(v):016x}" for v in snap.theta_d))
            y_extra += 44
        if snap.round_constant is not None:
            p.setPen(C_UNCH_ONE)
            p.drawText(12, y_extra, f"RC[{snap.round_index_abs}] = {snap.round_constant:016x}   (XORed into A[0,0])")
            y_extra += 22
        changed_lanes = int((snap.state != prev.state).sum()) if snap.step != "initial" else 0
        changed_bits = K.hamming_weight(snap.state ^ prev.state) if snap.step != "initial" else 0
        p.setPen(C_DIM)
        p.drawText(12, y_extra + 4, f"this step changed {changed_lanes} lanes / {changed_bits} bits;  "
                                    f"state Hamming weight {K.hamming_weight(snap.state)} / 1600")
