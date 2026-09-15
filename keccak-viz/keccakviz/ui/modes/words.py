"""Per-word view: the 1600-bit state as words of a configurable size (1 … 64
bits), one lane per row.  Highlights the words changed by the current step and
the words arriving on the bus during the loading phase."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import keccak as K
from .base import C_DIM, C_SEL, C_TEXT, C_TO_ONE, MONO, ModeWidget


class WordsView(ModeWidget):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.setFont(MONO)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(12, 50, 12, 0)
        bar.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        bar.addWidget(QtWidgets.QLabel("word size"))
        self.size_combo = QtWidgets.QComboBox()
        for w in session.WORD_SIZES:
            self.size_combo.addItem(f"{w} bits", w)
        self.size_combo.setCurrentIndex(list(session.WORD_SIZES).index(session.params.word_bits))
        self.size_combo.currentIndexChanged.connect(
            lambda i: session.set_params(word_bits=self.size_combo.itemData(i)))
        bar.addWidget(self.size_combo)
        self.binary = QtWidgets.QCheckBox("binary instead of hex")
        self.binary.toggled.connect(lambda _: self.update())
        bar.addWidget(self.binary)
        bar.addStretch(1)
        self.setLayout(bar)
        session.styleChanged.connect(lambda *_: self.update())

    def on_run_changed(self) -> None:
        wb = self.session.params.word_bits
        i = list(self.session.WORD_SIZES).index(wb)
        if self.size_combo.currentIndex() != i:
            self.size_combo.blockSignals(True)
            self.size_combo.setCurrentIndex(i)
            self.size_combo.blockSignals(False)
        self.update()

    # ------------------------------------------------------------ geometry

    def _geom(self):
        wb = self.session.params.word_bits
        per_lane = 64 // wb
        top, x0, rowh = 100, 150, 20
        digits = max(1, -(-wb // 4))
        cw = max(digits + 1, 3) * 8 + 8
        if self.binary.isChecked():
            cw = (wb + 1) * 7 + 8
        return wb, per_lane, top, x0, rowh, cw

    def _word_at(self, px, py):
        wb, per_lane, top, x0, rowh, cw = self._geom()
        lane = (py - top) // rowh
        col = (px - x0) // cw
        if 0 <= lane < 25 and 0 <= col < per_lane:
            return int(lane) * per_lane + int(col)
        return None

    def cell_at(self, px, py):
        w = self._word_at(px, py)
        if w is None:
            return None
        return K.bit_coords(w * self.session.params.word_bits)

    # ------------------------------------------------------------ paint

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        snap = s.snapshot
        prev = s.prev_snapshot
        wb, per_lane, top, x0, rowh, cw = self._geom()
        n_words = 1600 // wb
        rate_words = s.params.rate_bytes * 8 // wb
        self.draw_header(p, f"Words ({wb}-bit) — {s.position_text()}",
                         f"{n_words} words of {wb} bits in bit-string order (word w = bits {wb}·w … {wb}·w+{wb - 1}); "
                         f"one lane per row.  Words 0 … {rate_words - 1} are the rate.  "
                         f"The bus delivers {s.params.words_per_cycle} word(s) per load cycle.")
        cur = K.state_to_flat_bits(snap.state)
        prv = K.state_to_flat_bits(prev.state)
        loading = set(snap.info.get("words", [])) if snap.step == "load" and snap.info else set()
        output = set()
        if snap.step == "squeeze" and snap.info:
            output = set(range(-(-snap.info.get("nbits", 0) // wb)))
        f = QtGui.QFont(MONO)
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(C_DIM)
        for c in range(per_lane):
            p.drawText(x0 + c * cw, top - 6, f"+{c}")
        sel_word = None
        if s.selected:
            sel_word = K.bit_index(*s.selected) // wb
        hover_word = self._word_at(*self._hover_px) if getattr(self, "_hover_px", None) else None
        for lane in range(25):
            x, y = lane % 5, lane // 5
            yy = top + lane * rowh
            p.setPen(C_DIM)
            p.drawText(12, yy + 14, f"lane {lane:2d} A[{x},{y}]  w{lane * per_lane}")
            for c in range(per_lane):
                w = lane * per_lane + c
                bits = cur[w * wb:(w + 1) * wb]
                pbits = prv[w * wb:(w + 1) * wb]
                val = int(sum(int(b) << i for i, b in enumerate(bits)))
                changed = snap.step != "initial" and not (bits == pbits).all()
                r = QtCore.QRect(x0 + c * cw, yy, cw - 4, rowh - 3)
                bg = QtGui.QColor(50, 45, 40) if changed else (QtGui.QColor(34, 44, 40) if w < rate_words
                                                                 else QtGui.QColor(44, 34, 40))
                p.fillRect(r, bg)
                if w in loading:
                    p.setPen(QtGui.QPen(QtGui.QColor(80, 240, 255), 2))
                    p.drawRect(r)
                if w in output:
                    p.setPen(QtGui.QPen(QtGui.QColor(115, 255, 115), 2))
                    p.drawRect(r)
                if w == sel_word:
                    p.setPen(QtGui.QPen(C_SEL, 2))
                    p.drawRect(r)
                elif w == hover_word:
                    p.setPen(QtGui.QPen(C_TEXT, 1))
                    p.drawRect(r)
                p.setPen(C_TO_ONE if changed else C_TEXT)
                if self.binary.isChecked():
                    txt = "".join(str(int(b)) for b in bits[::-1])
                else:
                    txt = f"{val:0{max(1, -(-wb // 4))}x}"
                p.drawText(r.adjusted(4, 0, 0, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, txt)
        yb = top + 25 * rowh + 18
        p.setPen(C_DIM)
        info = ("green background = rate words, purple = capacity words, orange = changed by this step, "
                "cyan outline = arriving on the bus, green outline = read out as output")
        p.drawText(12, yb, info)
        w = hover_word if hover_word is not None else sel_word
        if w is not None:
            x, y, z = K.bit_coords(w * wb)
            p.setPen(C_TEXT)
            p.drawText(12, yb + 20, f"word {w}: bits {w * wb} … {(w + 1) * wb - 1} = lane {x + 5 * y} A[{x},{y}] "
                                    f"z {z} … {z + wb - 1}" if wb <= 64 else f"word {w}")

    def mouseMoveEvent(self, ev):
        self._hover_px = (ev.x(), ev.y())
        super().mouseMoveEvent(ev)
