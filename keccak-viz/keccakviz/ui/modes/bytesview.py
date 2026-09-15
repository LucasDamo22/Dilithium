"""Raw state bytes: the 200-byte buffer with the rate / capacity boundary."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui

from ...core import keccak as K
from .base import C_CAP, C_DIM, C_PAD, C_RATE, C_SEL, C_TEXT, C_TO_ONE, MONO, ModeWidget


class BytesView(ModeWidget):
    PER_ROW = 8  # one lane per row

    def _geom(self):
        w = self.width()
        bw = max(22, min(34, (w - 700) // self.PER_ROW))
        return 76, 250, bw, 22

    def cell_at(self, px, py):
        top, x0, bw, rh = self._geom()
        col = (px - x0) // bw
        row = (py - top) // rh
        if 0 <= col < 8 and 0 <= row < 25:
            lane = int(row)
            return lane % 5, lane // 5, int(col) * 8
        return None

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        snap = s.snapshot
        v = s.params
        rate = v.rate_bytes
        self.draw_header(p, f"State bytes — {s.position_text()}",
                         f"200 bytes, one lane (8 bytes, little-endian) per row.  Rate r = {rate} B ({rate * 8} bits) "
                         f"is what a block touches; capacity c = {200 - rate} B ({(200 - rate) * 8} bits) never meets "
                         f"the outside → {(200 - rate) * 4}-bit security.")
        data = K.state_to_bytes(snap.state)
        prev = K.state_to_bytes(s.prev_snapshot.state)
        block = None
        if snap.step == "initial" and s.perm.phase == "absorb" and s.perm.block_index < len(s.run.absorb_blocks):
            ab = s.run.absorb_blocks[s.perm.block_index]
            block = ab.data
            before = K.state_to_bytes(ab.state_before)
        top, x0, bw, rh = self._geom()
        f = QtGui.QFont(MONO)
        f.setPointSize(max(7, min(11, bw // 3)))
        p.setFont(f)
        for i in range(200):
            row, col = divmod(i, 8)
            r = QtCore.QRect(x0 + col * bw, top + row * rh, bw - 2, rh - 2)
            bg = C_RATE if i < rate else C_CAP
            p.fillRect(r, bg.darker(160))
            changed = data[i] != prev[i] and snap.step != "initial"
            if block is not None and i < len(block) and block[i] != 0 and data[i] != before[i]:
                p.fillRect(r, C_PAD.darker(140))
            if snap.step == "squeeze" and snap.info:
                a0 = snap.info.get("start_bit", 0) // 8
                if a0 <= i < a0 + snap.info.get("nbits", 0) // 8:
                    p.fillRect(r, QtGui.QColor(40, 110, 45))
            p.setPen(C_TO_ONE if changed else C_TEXT)
            p.drawText(r, QtCore.Qt.AlignCenter, f"{data[i]:02x}")
        for row in range(25):
            p.setPen(C_DIM)
            p.drawText(12, top + row * rh + 15, f"byte {row * 8:3d}   lane {row:2d} = A[{row % 5},{row // 5}]")
        # boundary
        by = top + (rate // 8) * rh - 1
        p.setPen(QtGui.QPen(QtGui.QColor(255, 230, 90), 3))
        p.drawLine(x0 - 4, by, x0 + 8 * bw, by)
        p.setPen(QtGui.QColor(255, 230, 90))
        p.drawText(x0 + 8 * bw + 10, by + 5, f"← rate/capacity boundary at byte {rate}")
        p.setPen(C_RATE.lighter(170))
        p.drawText(x0 + 8 * bw + 10, top + 15, f"rate: bytes 0…{rate - 1}  (absorb XORs here, squeeze reads here)")
        p.setPen(C_CAP.lighter(170))
        p.drawText(x0 + 8 * bw + 10, top + 24 * rh + 15, f"capacity: bytes {rate}…199  (only the permutation touches these)")
        if block is not None:
            p.setPen(C_PAD)
            p.drawText(x0 + 8 * bw + 10, top + 40, "orange: bytes just XORed in by this absorb block (non-zero)")
        for ci, (tx, ty, tz) in s.tracked_at():
            row, col = tx + 5 * ty, tz // 8
            colr = QtGui.QColor(s.TRACK_COLORS[ci])
            p.setPen(QtGui.QPen(colr, 2))
            p.drawRect(QtCore.QRect(x0 + col * bw, top + row * rh, bw - 2, rh - 2))
            p.setPen(colr)
            p.drawText(x0 + 8 * bw + 10 + 480, top + row * rh + 15, f"#{ci + 1} bit {tz % 8} of this byte")
        legend_y = top + 25 * rh + 18
        p.setPen(C_DIM)
        p.drawText(12, legend_y, "bit i of the 1600-bit state string = bit (i mod 8) of byte i div 8;  "
                                 "lane (x,y) is bytes 8·(x+5y) … 8·(x+5y)+7, least significant byte first.")
        cell = s.selected or self._hover
        if cell:
            self.draw_cell_info(p, cell, legend_y + 20)
            if s.selected:
                x, y, _ = s.selected
                row = x + 5 * y
                p.setPen(QtGui.QPen(C_SEL, 2))
                p.drawRect(QtCore.QRect(x0 - 2, top + row * rh - 2, 8 * bw + 2, rh + 2))
