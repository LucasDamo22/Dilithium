"""Diffusion heatmap: for each state bit, the round at which it first
depended on one chosen input bit."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import keccak as K
from .base import C_DIM, C_TEXT, ModeWidget
from .grid import draw_lane_grid, grid_cell_at, grid_geometry, viridis


class CellPicker(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal(tuple)

    def __init__(self, label: str, initial=(0, 0, 0), parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel(label))
        self.spins = []
        for name, maxv, v in (("x", 4, initial[0]), ("y", 4, initial[1]), ("z", 63, initial[2])):
            lay.addWidget(QtWidgets.QLabel(name))
            sp = QtWidgets.QSpinBox()
            sp.setRange(0, maxv)
            sp.setValue(v)
            sp.valueChanged.connect(self._emit)
            lay.addWidget(sp)
            self.spins.append(sp)

    def value(self):
        return tuple(sp.value() for sp in self.spins)

    def set_value(self, cell) -> None:
        for sp, v in zip(self.spins, cell):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def _emit(self, _v) -> None:
        self.changed.emit(self.value())


class DiffusionHeatmap(ModeWidget):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(12, 50, 12, 0)
        bar.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.picker = CellPicker("source bit", session.params.diffusion_source)
        self.picker.changed.connect(lambda c: session.set_params(diffusion_source=c))
        bar.addWidget(self.picker)
        self.use_sel = QtWidgets.QPushButton("use selected cell")
        self.use_sel.clicked.connect(self._use_selected)
        bar.addWidget(self.use_sel)
        self.upto = QtWidgets.QCheckBox("only up to the current step")
        self.upto.toggled.connect(lambda _: self.update())
        bar.addWidget(self.upto)
        bar.addStretch(1)
        self.setLayout(bar)

    def _use_selected(self) -> None:
        if self.session.selected:
            self.picker.set_value(self.session.selected)
            self.session.set_params(diffusion_source=self.session.selected)

    def _geom(self):
        return grid_geometry(self.width(), self.height(), 120, 90)

    def cell_at(self, px, py):
        x0, y0, cw, ch = self._geom()
        return grid_cell_at(px, py, x0, y0, cw, ch)

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        d = s.diffusion()
        n = s.params.num_rounds
        self.draw_header(p, f"Diffusion heatmap — permutation {s.perm_index + 1}, flipping input bit "
                            f"(x={d.source[0]}, y={d.source[1]}, z={d.source[2]})",
                         "Colour = round (fractional: +0.2 per step mapping) at which each bit first differed between the two runs. "
                         "Every bit is reached after a few rounds; without θ it takes far longer.")
        x0, y0, cw, ch = self._geom()
        limit = s.snap_index if self.upto.isChecked() else 10 ** 9
        fr = d.first_round()

        def color_of(x, y, z):
            fs = int(d.first_step[x, y, z])
            if fs < 0 or fs > limit:
                return QtGui.QColor(40, 42, 50)
            if fs == 0:
                return QtGui.QColor(255, 255, 255)
            return viridis(1.0 - min(1.0, fr[x, y, z] / max(1.0, min(n, 6))))

        draw_lane_grid(p, x0, y0, cw, ch, color_of, s.selected, self._hover)
        # legend
        ly = int(y0 + 25 * ch + 14)
        p.setPen(C_DIM)
        p.drawText(x0, ly + 12, "first affected in round:")
        for i in range(60):
            t = i / 59
            p.fillRect(QtCore.QRectF(x0 + 170 + i * 3, ly, 3, 14), viridis(1.0 - t))
        p.drawText(x0 + 170, ly + 28, "1 (right after the flip)")
        p.drawText(x0 + 170 + 180 - 40, ly + 28, f"{min(n, 6)}+")
        p.fillRect(QtCore.QRectF(x0 + 400, ly, 14, 14), QtGui.QColor(255, 255, 255))
        p.drawText(x0 + 420, ly + 12, "the flipped bit itself")
        p.fillRect(QtCore.QRectF(x0 + 560, ly, 14, 14), QtGui.QColor(40, 42, 50))
        p.drawText(x0 + 580, ly + 12, "not yet affected")
        # counts per round
        counts = [d.reached_by_round(r) for r in range(1, n + 1)]
        txt = "bits affected after round: " + "  ".join(f"r{r + 1}:{c}" for r, c in enumerate(counts[:8]))
        p.setPen(C_TEXT)
        p.drawText(x0, ly + 48, txt)
        cell = s.selected or self._hover
        if cell:
            x, y, z = cell
            fs = int(d.first_step[x, y, z])
            when = "never (within traced rounds)" if fs < 0 else (
                "it is the source" if fs == 0 else f"snapshot {fs}: {s.trace[fs].label}")
            p.drawText(x0, ly + 66, f"cell ({x},{y},{z}): first affected at {when}")


__all__ = ["DiffusionHeatmap", "CellPicker"]
