"""Slice stack: the 64 x-y slices unrolled into a grid.  Theta and chi act
inside a slice, so this view makes their structure obvious.  Step
animations run here too, on the shared clock: rho flies cells between
slices, pi shuffles them inside every slice, the flip steps pulse."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui

from ...core import keccak as K
from .. import anim
from PyQt5 import QtWidgets

from .base import C_DIM, C_SEL, C_SRC, C_TEXT, WORD_TINT, ModeWidget, qcolor


class SliceStack(ModeWidget):
    def __init__(self, session, animator=None, parent=None):
        super().__init__(session, parent)
        self.animator = animator
        self.COLS, self.ROWS = 16, 4
        if animator is not None:
            animator.changed.connect(self.update)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(12, 50, 12, 0)
        bar.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.words_cb = QtWidgets.QCheckBox("words")
        self.words_cb.setToolTip("Tint alternate words and outline the selected cell's word")
        self.words_cb.setChecked(session.show_words)
        self.words_cb.toggled.connect(session.set_show_words)
        session.wordsChanged.connect(lambda on: self.words_cb.setChecked(on))
        session.wordsChanged.connect(lambda _on: self.update())
        bar.addWidget(self.words_cb)
        bar.addWidget(QtWidgets.QLabel("word size"))
        self.size_combo = QtWidgets.QComboBox()
        for w in session.WORD_SIZES:
            self.size_combo.addItem(f"{w} bits", w)
        self.size_combo.setCurrentIndex(list(session.WORD_SIZES).index(session.params.word_bits))
        self.size_combo.currentIndexChanged.connect(
            lambda i: session.set_params(word_bits=self.size_combo.itemData(i)))
        bar.addWidget(self.size_combo)
        bar.addStretch(1)
        self.setLayout(bar)
        session.styleChanged.connect(lambda *_: self.update())
        session.visibilityChanged.connect(lambda *_: self.update())
        session.dyesChanged.connect(self.update)

    # ------------------------------------------------------------ geometry

    def on_run_changed(self) -> None:
        i = list(self.session.WORD_SIZES).index(self.session.params.word_bits)
        if self.size_combo.currentIndex() != i:
            self.size_combo.blockSignals(True)
            self.size_combo.setCurrentIndex(i)
            self.size_combo.blockSignals(False)
        self.update()

    def _layout(self):
        w, h = self.width(), self.height()
        top = 80
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

    def cell_origin(self, x, y, z, geom=None) -> Tuple[float, float]:
        """Top-left pixel of cell (x, y, z); x/y/z may be fractional (mid-animation)."""
        top, cell, sw, sh, x0 = geom or self._layout()
        zi = int(np.floor(z + 1e-9))
        col, row = zi % self.COLS, zi // self.COLS
        sx = x0 + col * sw + 4
        sy = top + row * sh + 14
        return sx + x * cell, sy + (4 - y) * cell

    def cell_rect(self, x: int, y: int, z: int) -> QtCore.QRectF:
        top, cell, sw, sh, x0 = self._layout()
        ox, oy = self.cell_origin(x, y, z, (top, cell, sw, sh, x0))
        return QtCore.QRectF(ox, oy, cell, cell)

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

    # ------------------------------------------------------------ frame

    def _frame(self):
        """(prev snapshot, snapshot, t, Frame) on the shared clock."""
        s = self.session
        if self.animator is not None:
            prev, snap, t = self.animator.pair()
        else:
            prev, snap, t = s.prev_snapshot, s.snapshot, 1.0
        mode, style = s.color_mode, s.cell_style
        prev_bits = K.lanes_to_bits(prev.state).reshape(-1)
        cur_bits = K.lanes_to_bits(snap.state).reshape(-1)
        diff = prev_diff = None
        if mode == "avalanche":
            av = s.avalanche()
            diff = av.diff_bits(s.core_index(snap.index)).reshape(-1)
            prev_diff = av.diff_bits(s.core_index(prev.index)).reshape(-1)
        dye_prev = dye_cur = None
        if mode == "dye":
            dye_prev, dye_cur = s.dye_render(prev.index), s.dye_render(snap.index)
        prev_prev = s.trace[max(0, prev.index - 1)]
        prev_colors = anim.static_colors(prev_bits, K.lanes_to_bits(prev_prev.state).reshape(-1), mode, prev_diff,
                                         style, dye_prev)
        cur_colors = anim.static_colors(cur_bits, prev_bits, mode, diff, style, dye_cur)
        load_cells = None
        if snap.step == "load" and snap.info and snap.info.get("cells"):
            load_cells = np.array([anim.cell_index(*K.bit_coords(i)) for i in snap.info["cells"]])
        fr = anim.build_frame(snap.step, prev_bits, cur_bits, t, mode, diff, snap.skipped,
                              snap.theta_c, snap.theta_d, False, prev_colors, style, cur_colors, load_cells,
                              show_bus=False)
        return prev, snap, t, fr

    # ------------------------------------------------------------ paint

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        prev, snap, t, fr = self._frame()
        transition = t < 1.0 and snap.step not in ("initial",) and not snap.skipped
        self.draw_header(p, f"Slice stack — {s.position_text()}",
                         "64 slices z = 0 … 63, each a 5×5 grid with x to the right and y up.  "
                         "θ and χ never leave a slice; ρ moves bits between slices; π shuffles lanes within every slice.")
        geom = self._layout()
        top, cell, sw, sh, x0 = geom
        f = p.font()
        f.setPointSize(max(6, min(9, cell)))
        p.setFont(f)
        anchor = s.selected or self._hover
        struct_mask = None
        sc = None
        if s.structure and anchor:
            struct_mask = anim.structure_cells(s.structure, *anchor)
            sc = qcolor(anim.STRUCTURE_COLORS[s.structure])
        srcs = set()
        nxt = None
        if s.selected and s.snap_index + 1 < s.num_snapshots and not transition:
            nxt = s.trace[s.snap_index + 1].step
            srcs = set(K.sources_of(nxt, *s.selected))
        for z in range(64):
            col, row = z % self.COLS, z // self.COLS
            p.setPen(C_DIM)
            p.drawText(QtCore.QPointF(x0 + col * sw + 4, top + row * sh + 11), f"z={z}")
            p.fillRect(QtCore.QRectF(x0 + col * sw + 4, top + row * sh + 14, cell * 5, cell * 5),
                       QtGui.QColor(30, 32, 40))

        # cells: screen position interpolated between the previous and the destination cell
        step = snap.step
        moving = transition and step in ("rho", "pi")
        if moving:
            u = anim.smoothstep(t)
            if step == "rho":
                dz = K.RHO_OFFSETS[anim.XS, anim.YS]
                dx, dy = anim.XS, anim.YS
                dz = (anim.ZS + dz) % 64
            else:
                dx, dy = anim.YS, (2 * anim.XS + 3 * anim.YS) % 5
                dz = anim.ZS
        _s1 = anim.STYLES[s.cell_style][2]
        if s.show_words:
            wb = max(1, s.params.word_bits)
            odd = (K.bit_index(anim.XS, anim.YS, anim.ZS) // wb) % 2 == 1
            if s.color_mode == "dye":
                fr.col[:1600][odd] *= 0.82  # keep dye hues intact
            else:
                fr.col[:1600][odd] = np.clip(fr.col[:1600][odd] * WORD_TINT, 0, 1)
        layout_k = prev.index if transition else s.snap_index
        big = s.tracked_big(layout_k)
        if big:
            idx = np.array([anim.cell_index(*c) for _ci, c in big])
            cols = np.array([[QtGui.QColor(s.TRACK_COLORS[ci]).getRgbF()[j] for j in range(3)] for ci, _c in big])
            fr.col[idx] = fr.col[idx] * 0.35 + cols * 0.65
            fr.scale[idx] = np.maximum(fr.scale[idx], 0.45)
        if s.visibility != "both":
            layout_bits = K.lanes_to_bits((prev if transition else snap).state).reshape(-1)
            fr.scale[:1600][layout_bits == (1 if s.visibility == "zeros" else 0)] = 0.0
        order = np.arange(1600)
        if moving:
            order = np.argsort(fr.scale)  # draw the big (moving, in-flight) ones last
        for i in order:
            x, y, z = int(anim.XS[i]), int(anim.YS[i]), int(anim.ZS[i])
            scale = float(fr.scale[i])
            if scale <= 0.02:
                continue
            ox, oy = self.cell_origin(x, y, z, geom)
            if moving:
                tx, ty = self.cell_origin(int(dx[i]), int(dy[i]), int(dz[i]), geom)
                ox, oy = ox + (tx - ox) * u, oy + (ty - oy) * u
            size = cell * max(0.25, min(1.0, scale / max(_s1, 0.01)))
            pad = (cell - size) / 2
            r = QtCore.QRectF(ox + pad, oy + pad, size, size)
            c = qcolor(fr.col[i], 0.6 if (fr.alpha is not None and fr.alpha[i] < 1) else 1.0)
            if struct_mask is not None:
                if struct_mask[i]:
                    c = QtGui.QColor((c.red() + sc.red()) // 2, (c.green() + sc.green()) // 2,
                                     (c.blue() + sc.blue()) // 2)
                else:
                    c = c.darker(180)
            p.fillRect(r.adjusted(0.5, 0.5, -0.5, -0.5), c)
            if (x, y, z) in srcs:
                p.setPen(QtGui.QPen(C_SRC, 2))
                p.drawRect(QtCore.QRectF(ox, oy, cell, cell).adjusted(1, 1, -1, -1))
        if s.structure == "slice" and anchor:
            z = anchor[2]
            col, row = z % self.COLS, z // self.COLS
            p.setPen(QtGui.QPen(sc, 2))
            p.drawRect(QtCore.QRectF(x0 + col * sw + 3, top + row * sh + 13, cell * 5 + 2, cell * 5 + 2))

        # tracked bits (small groups) follow their cell mid-flight
        for ci, (x, y, z), _t in s.tracked_small(layout_k):
            ox, oy = self.cell_origin(x, y, z, geom)
            if moving:
                i = anim.cell_index(x, y, z)
                tx, ty = self.cell_origin(int(dx[i]), int(dy[i]), int(dz[i]), geom)
                ox, oy = ox + (tx - ox) * u, oy + (ty - oy) * u
            colr = QtGui.QColor(s.TRACK_COLORS[ci])
            p.setPen(QtGui.QPen(colr, 2))
            p.drawRect(QtCore.QRectF(ox, oy, cell, cell))
            if cell >= 14:
                p.setPen(colr)
                p.drawText(QtCore.QPointF(ox + cell + 2, oy + 8), f"#{ci + 1}")
        if s.selected and s.show_words and not transition:
            wi, b0, b1 = s.word_of(s.selected)
            p.setPen(QtGui.QPen(QtGui.QColor(140, 205, 255), 2))
            for i in range(b0, b1 + 1):
                p.drawRect(self.cell_rect(*K.bit_coords(i)))
            p.setPen(QtGui.QColor(140, 205, 255))
            p.drawText(12, self.height() - 62, f"word {wi}: bits {b0}…{b1} ({s.params.word_bits}-bit words), outlined in blue")
        if s.selected and not transition:
            p.setPen(QtGui.QPen(C_SEL, 2))
            p.drawRect(self.cell_rect(*s.selected))
        if self._hover and self._hover != s.selected:
            p.setPen(QtGui.QPen(C_TEXT, 1))
            p.drawRect(self.cell_rect(*self._hover))
        if snap.step == "load" and snap.info and snap.info.get("cells"):
            # outline the words arriving in this bus cycle
            p.setPen(QtGui.QPen(QtGui.QColor(80, 240, 255), 2))
            for i in snap.info["cells"]:
                x, y, z = K.bit_coords(i)
                ox, oy = self.cell_origin(x, y, z, geom)
                p.drawRect(QtCore.QRectF(ox, oy, cell, cell))
        if transition and step in ("rho", "pi"):
            p.setPen(C_DIM)
            p.drawText(12, self.height() - 46, "ρ: cells fly to slice z + r[x,y] (wrapping at 64);  "
                                                "π: cells move inside their slice to (y, 2x+3y).  Ghosted cells are passing through others.")
        cell_info = s.selected or self._hover
        if cell_info:
            self.draw_cell_info(p, cell_info, self.height() - 14)
            if nxt and s.selected:
                p.setPen(C_SRC)
                p.drawText(12, self.height() - 30,
                           f"cyan outline: the {len(srcs)} cells that feed the selected cell in the next step ({nxt})")
