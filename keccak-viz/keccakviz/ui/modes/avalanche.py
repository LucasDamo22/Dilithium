"""Avalanche view: two runs differing in one input bit, the XOR difference
at the current step, and the Hamming distance per round."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import keccak as K
from .base import C_DIFF, C_DIM, C_TEXT, ModeWidget
from .grid import draw_lane_grid, grid_cell_at, grid_geometry
from .heatmap import CellPicker


class AvalancheView(ModeWidget):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(12, 50, 12, 0)
        bar.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.picker = CellPicker("flip input bit", session.params.avalanche_flip)
        self.picker.changed.connect(lambda c: session.set_params(avalanche_flip=c))
        bar.addWidget(self.picker)
        b = QtWidgets.QPushButton("use selected cell")
        b.clicked.connect(self._use_selected)
        bar.addWidget(b)
        b = QtWidgets.QPushButton("show difference on the 3D cube")
        b.setToolTip("switch the cube colour mode to 'avalanche difference'")
        b.clicked.connect(lambda: session.set_color_mode("avalanche"))
        bar.addWidget(b)
        self.per_step = QtWidgets.QCheckBox("plot every step mapping")
        self.per_step.toggled.connect(lambda _: self.update())
        bar.addWidget(self.per_step)
        bar.addStretch(1)
        self.setLayout(bar)

    def _use_selected(self) -> None:
        if self.session.selected:
            self.picker.set_value(self.session.selected)
            self.session.set_params(avalanche_flip=self.session.selected)

    def _geom(self):
        h = self.height()
        plot_h = max(150, int(h * 0.32))
        x0, y0, cw, ch = grid_geometry(self.width(), h, 120, plot_h + 40)
        return x0, y0, cw, ch, plot_h

    def cell_at(self, px, py):
        x0, y0, cw, ch, _ = self._geom()
        return grid_cell_at(px, py, x0, y0, cw, ch)

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        av = s.avalanche()
        snap = s.snapshot
        fx, fy, fz = s.params.avalanche_flip
        diff = av.diff_bits(snap.index)
        hd = int(diff.sum())
        enabled = ", ".join(s.params.enabled_steps) if len(s.params.enabled_steps) < 5 else "all five step mappings"
        self.draw_header(p, f"Avalanche — {s.position_text()}   ·   Hamming distance now: {hd} / 1600",
                         f"Run A: the real state.  Run B: the same state with input bit (x={fx}, y={fy}, z={fz}) flipped.  "
                         f"Magenta = bits that differ.  Enabled steps: {enabled}.")
        x0, y0, cw, ch, plot_h = self._geom()
        cur = K.lanes_to_bits(snap.state)

        def color_of(x, y, z):
            if diff[x, y, z]:
                return C_DIFF
            return QtGui.QColor(70, 62, 48) if cur[x, y, z] else QtGui.QColor(40, 42, 50)

        draw_lane_grid(p, x0, y0, cw, ch, color_of, s.selected, self._hover)
        self.draw_tracked(p, lambda x, y, z: QtCore.QRectF(x0 + z * cw, y0 + (x + 5 * y) * ch, cw, ch))
        # ---- plot
        top = int(y0 + 25 * ch + 30)
        left, right = x0, int(x0 + 64 * cw)
        bottom = int(top + plot_h - 40)
        p.fillRect(QtCore.QRectF(left, top, right - left, bottom - top), QtGui.QColor(30, 32, 40))
        per_step = self.per_step.isChecked()
        ys = av.hamming_per_snapshot() if per_step else av.hamming_per_round()
        n = len(ys) - 1
        xs_lbl = "step mapping index" if per_step else "round"

        def px_(i):
            return left + (right - left) * i / max(1, n)

        def py_(v):
            return bottom - (bottom - top) * v / 1600.0

        p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 130), 1, QtCore.Qt.DashLine))
        p.drawLine(QtCore.QPointF(left, py_(800)), QtCore.QPointF(right, py_(800)))
        p.setPen(C_DIM)
        p.drawText(QtCore.QPointF(right + 6, py_(800) + 4), "800 = half the state (ideal)")
        p.drawText(QtCore.QPointF(right + 6, py_(1600) + 4), "1600")
        p.drawText(QtCore.QPointF(right + 6, py_(0) + 4), "0")
        p.drawText(QtCore.QPointF(left, bottom + 30), f"Hamming distance between the two runs vs {xs_lbl}")
        for r in range(0, n + 1, max(1, n // 12)):
            p.drawText(QtCore.QPointF(px_(r) - 4, bottom + 14), str(r))
        path = QtGui.QPainterPath()
        for i, v in enumerate(ys):
            pt = QtCore.QPointF(px_(i), py_(v))
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(QtGui.QPen(C_DIFF, 2))
        p.drawPath(path)
        for i, v in enumerate(ys):
            p.drawEllipse(QtCore.QPointF(px_(i), py_(v)), 2.5, 2.5)
        # current position marker
        cur_i = snap.index if per_step else (0 if snap.step == "initial" else snap.round + (1 if snap.step == "iota" else 0))
        if not per_step and snap.step not in ("initial", "iota"):
            cur_i = snap.round + (K.STEP_NAMES.index(snap.step) + 1) / 5.0
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        p.drawLine(QtCore.QPointF(px_(cur_i), top), QtCore.QPointF(px_(cur_i), bottom))
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        p.setPen(C_TEXT)
        rounds = av.hamming_per_round()
        p.drawText(QtCore.QPointF(left, bottom + 48),
                   "per round: " + "  ".join(f"r{i}:{int(v)}" for i, v in enumerate(rounds[: min(len(rounds), 13)])))
        if len(s.params.enabled_steps) < 5:
            p.setPen(QtGui.QColor(255, 150, 90))
            p.drawText(QtCore.QPointF(left, bottom + 66),
                       "some step mappings are disabled - compare this curve with the full permutation")
