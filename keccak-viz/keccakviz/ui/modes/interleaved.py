"""Bit-interleaved layout: each 64-bit lane as two 32-bit words (even / odd
bit positions), side by side with the plain layout, and a worked example of
how one 64-bit rotation becomes two 32-bit rotations."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui

from ...core import keccak as K
from ...core import layouts as L
from .base import C_DIM, C_TEXT, MONO, ModeWidget

C_EVEN = QtGui.QColor(80, 200, 230)
C_ODD = QtGui.QColor(190, 130, 255)


class InterleavedView(ModeWidget):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.setFont(MONO)

    def _lane(self):
        sel = self.session.selected or self._hover
        if sel:
            return sel[0], sel[1]
        return 1, 0  # a lane with an odd offset (r = 1) by default: shows the swap

    def cell_at(self, px, py):
        top, rowh = 100, 19
        lane = (py - top) // rowh
        if 0 <= lane < 25 and 12 <= px < 700:
            return int(lane) % 5, int(lane) // 5, 0
        return None

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        snap = s.snapshot
        self.draw_header(p, f"Bit-interleaved layout — {s.position_text()}",
                         "A 64-bit lane L is split into E = even bits (0,2,4,…) and O = odd bits (1,3,5,…), each a 32-bit word.  "
                         "XOR/AND/NOT don't care; only rotations do - and ROL64 by r becomes two ROL32s (with a swap when r is odd).")
        f = QtGui.QFont(MONO)
        f.setPointSize(9)
        p.setFont(f)
        top, rowh = 100, 19
        p.setPen(C_DIM)
        p.drawText(12, top - 8, "lane        64-bit (plain)          E = even bits    O = odd bits    rho r   ROL64 r  ->  E'            O'")
        even, odd = L.state_interleaved(snap.state)
        lx, ly = self._lane()
        for lane in range(25):
            x, y = lane % 5, lane // 5
            v = int(snap.state[x, y])
            e, o = int(even[x, y]), int(odd[x, y])
            r = int(K.RHO_OFFSETS[x, y])
            plan = L.interleave_rotation_plan(r)
            yy = top + lane * rowh + 14
            if (x, y) == (lx, ly):
                p.fillRect(QtCore.QRect(8, yy - 14, 900, rowh), QtGui.QColor(60, 62, 75))
            p.setPen(C_TEXT)
            p.drawText(12, yy, f"A[{x},{y}]     {v:016x}")
            p.setPen(C_EVEN)
            p.drawText(12 + 32 * 8 + 10, yy, f"{e:08x}")
            p.setPen(C_ODD)
            p.drawText(12 + 49 * 8 + 10, yy, f"{o:08x}")
            p.setPen(C_DIM)
            p.drawText(12 + 65 * 8 + 10, yy, f"{r:2d}")
            p.setPen(C_TEXT)
            desc = (f"E'=ROL32({plan['even_from']},{plan['even_rot']:2d})  O'=ROL32({plan['odd_from']},{plan['odd_rot']:2d})"
                    + ("   swap" if plan["swap"] else ""))
            p.drawText(12 + 73 * 8 + 10, yy, desc)
        # ---- worked example for the highlighted lane
        v = int(snap.state[lx, ly])
        e, o = L.interleave(v)
        r = int(K.RHO_OFFSETS[lx, ly])
        e2, o2 = L.rol64_interleaved(e, o, r)
        v2 = int(K.rol64(snap.state[lx, ly:ly + 1], r)[0])
        plan = L.interleave_rotation_plan(r)
        bx = 12
        by = top + 25 * rowh + 30
        bw = max(6, min(14, (self.width() - 200) // 64))
        bh = bw
        f.setPointSize(10)
        p.setFont(f)
        p.setPen(C_TEXT)
        p.drawText(bx, by - 8, f"worked example on lane A[{lx},{ly}] (lane {lx + 5 * ly}), rho offset r = {r}: "
                               f"{'even r: both words rotate by r/2' if not plan['swap'] else 'odd r: the words swap, E gets O rotated by (r+1)/2, O gets E rotated by (r-1)/2'}")

        def bits_row(label, value, nbits, y0, col_fn, note=""):
            p.setPen(C_DIM)
            p.drawText(bx, y0 + bh - 2, label)
            for i in range(nbits):
                bit = (value >> i) & 1
                # draw MSB on the left like a hex dump: bit index decreasing to the right
                rx = bx + 90 + (nbits - 1 - i) * bw
                c = col_fn(i)
                r_ = QtCore.QRect(rx, y0, bw - 1, bh - 1)
                p.fillRect(r_, c if bit else c.darker(300))
            if note:
                p.setPen(C_DIM)
                p.drawText(bx + 90 + nbits * bw + 8, y0 + bh - 2, note)

        y0 = by + 8
        bits_row("L (64)", v, 64, y0, lambda i: C_EVEN if i % 2 == 0 else C_ODD, "bit 63 … bit 0; colour = which word the bit goes to")
        bits_row("E (32)", e, 32, y0 + bh + 6, lambda i: C_EVEN, "E[i] = L[2i]")
        bits_row("O (32)", o, 32, y0 + 2 * (bh + 6), lambda i: C_ODD, "O[i] = L[2i+1]")
        y1 = y0 + 3 * (bh + 6) + 14
        p.setPen(C_TEXT)
        p.drawText(bx, y1 + bh - 2, f"ROL64(L, {r}):")
        bits_row("L' (64)", v2, 64, y1 + bh + 6, lambda i: C_EVEN if i % 2 == 0 else C_ODD, f"= {v2:016x}")
        bits_row("E' (32)", e2, 32, y1 + 2 * (bh + 6), lambda i: C_EVEN,
                 f"= ROL32({plan['even_from']}, {plan['even_rot']}) = {e2:08x}")
        bits_row("O' (32)", o2, 32, y1 + 3 * (bh + 6), lambda i: C_ODD,
                 f"= ROL32({plan['odd_from']}, {plan['odd_rot']}) = {o2:08x}")
        ok = L.deinterleave(e2, o2) == v2
        p.setPen(QtGui.QColor(120, 255, 140) if ok else QtGui.QColor(255, 100, 100))
        p.drawText(bx, y1 + 4 * (bh + 6) + 16,
                   f"de-interleave(E', O') == ROL64(L, {r}) ?  {'yes' if ok else 'NO'}   "
                   "→ a 32-bit CPU never needs a 64-bit shifter; θ's ROT(C,1) is a swap plus one ROL32 by 1 of one word.")
        p.setPen(C_DIM)
        p.drawText(bx, y1 + 4 * (bh + 6) + 34, "click a lane in the table above (or a cell in any other view) to change the example")
