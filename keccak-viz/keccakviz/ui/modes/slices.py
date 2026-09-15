"""Slice stack: the 64 x-y slices unrolled into a grid.  Theta and chi act
inside a slice, so this view makes their structure obvious."""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui

from ...core import keccak as K
from .. import anim
from .base import C_DIM, C_SEL, C_SRC, C_TEXT, ModeWidget, bit_colors, qcolor


class SliceStack(ModeWidget):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.COLS, self.ROWS = 16, 4

    def _layout(self):
        w, h = self.width(), self.height()
        top = 56
        bottom = 40
        avail_w = w - 24
        avail_h = h - top - bottom
        best = None
        for cols in (16, 8, 32):
            rows = 64 // cols
            cell = int(max(3, min((avail_w / cols - 8) / 5, (avail_h / rows - 18) / 5)))
            if best is None or cell > best[0]:
                best = (cell, cols, rows)
        cell, cols, rows = best
        self.COLS, self.ROWS = cols, rows
        sw = cell * 5 + 8
        sh = cell * 5 + 18
        x0 = 12 + (avail_w - sw * cols) / 2
        return top, cell, sw, sh, x0

    def cell_rect(self, x: int, y: int, z: int) -> QtCore.QRectF:
        top, cell, sw, sh, x0 = self._layout()
        col, row = z % self.COLS, z // self.COLS
        sx = x0 + col * sw + 4
        sy = top + row * sh + 14
        # y = 0 at the bottom, like the 3D view
        return QtCore.QRectF(sx + x * cell, sy + (4 - y) * cell, cell, cell)

    def cell_at(self, px: int, py: int) -> Optional[Tuple[int, int, int]]:
        top, cell, sw, sh, x0 = self._layout()
        col = int((px - x0) // sw)
        row = int((py - top) // sh)
        if not (0 <= col < self.COLS and 0 <= row < self.ROWS):
            return None
        z = row * self.COLS + col
        sx = x0 + col * sw + 4
        sy = top + row * sh + 14
        x = int((px - sx) // cell)
        yi = int((py - sy) // cell)
        if 0 <= x < 5 and 0 <= yi < 5:
            return x, 4 - yi, z
        return None

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        snap = s.snapshot
        self.draw_header(p, f"Slice stack — {s.position_text()}",
                         "64 slices z = 0 … 63, each a 5×5 grid with x to the right and y up.  "
                         "θ and χ never leave a slice; ρ moves bits between slices; π shuffles lanes within every slice.")
        cols = bit_colors(s, snap)
        top, cell, sw, sh, x0 = self._layout()
        f = p.font()
        f.setPointSize(max(6, min(9, cell)))
        p.setFont(f)
        anchor = s.selected or self._hover
        struct_mask = None
        if s.structure and anchor:
            struct_mask = anim.structure_cells(s.structure, *anchor).reshape(5, 5, 64)
            sc = qcolor(anim.STRUCTURE_COLORS[s.structure])
        srcs = set()
        nxt = None
        if s.selected and s.snap_index + 1 < s.num_snapshots:
            nxt = s.trace[s.snap_index + 1].step
            srcs = set(K.sources_of(nxt, *s.selected))
        for z in range(64):
            col, row = z % self.COLS, z // self.COLS
            sx = x0 + col * sw
            sy = top + row * sh
            p.setPen(C_DIM)
            p.drawText(QtCore.QPointF(sx + 4, sy + 11), f"z={z}")
            for x in range(5):
                for y in range(5):
                    r = self.cell_rect(x, y, z)
                    c = qcolor(cols[x, y, z])
                    if struct_mask is not None:
                        if struct_mask[x, y, z]:
                            c = QtGui.QColor((c.red() + sc.red()) // 2, (c.green() + sc.green()) // 2,
                                             (c.blue() + sc.blue()) // 2)
                        else:
                            c = c.darker(180)
                    p.fillRect(r.adjusted(0.5, 0.5, -0.5, -0.5), c)
                    if (x, y, z) in srcs:
                        p.setPen(QtGui.QPen(C_SRC, 2))
                        p.drawRect(r.adjusted(1, 1, -1, -1))
            if s.structure == "slice" and anchor and anchor[2] == z:
                p.setPen(QtGui.QPen(sc, 2))
                p.drawRect(QtCore.QRectF(sx + 3, sy + 13, cell * 5 + 2, cell * 5 + 2))
        if s.selected:
            r = self.cell_rect(*s.selected)
            p.setPen(QtGui.QPen(C_SEL, 2))
            p.drawRect(r)
        if self._hover and self._hover != s.selected:
            r = self.cell_rect(*self._hover)
            p.setPen(QtGui.QPen(C_TEXT, 1))
            p.drawRect(r)
        cell_info = s.selected or self._hover
        if cell_info:
            self.draw_cell_info(p, cell_info, self.height() - 14)
            if nxt and s.selected:
                p.setPen(C_SRC)
                p.drawText(12, self.height() - 30, f"cyan outline: the {len(srcs)} cells that feed the selected cell in the next step ({nxt})")
