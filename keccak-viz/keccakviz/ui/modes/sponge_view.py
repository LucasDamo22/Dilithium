"""The sponge-level screen: message + padding, rate/capacity split, the
absorb/squeeze timeline and the permutation-call counter."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import sponge as S
from ..session import Session
from .base import C_BG, C_CAP, C_DIM, C_PAD, C_RATE, C_SEL, C_TEXT, MONO


class SpongeCanvas(QtWidgets.QWidget):
    """Custom-painted: padded message, rate/capacity bar, timeline."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setMouseTracking(True)
        self.setMinimumHeight(520)
        self._perm_rects: List[Tuple[QtCore.QRectF, int]] = []
        self._hover_perm: Optional[int] = None
        session.runChanged.connect(self.update)
        session.positionChanged.connect(lambda *_: self.update())

    def mouseMoveEvent(self, ev):
        h = None
        for r, i in self._perm_rects:
            if r.contains(ev.x(), ev.y()):
                h = i
        if h != self._hover_perm:
            self._hover_perm = h
            self.setCursor(QtCore.Qt.PointingHandCursor if h is not None else QtCore.Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, ev):
        for r, i in self._perm_rects:
            if r.contains(ev.x(), ev.y()):
                self.session.set_position(perm_index=i, snap_index=0)
                self.session.jump_to_cube.emit()
                return

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), C_BG)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p: QtGui.QPainter):
        s = self.session
        run = s.run
        v = run.variant
        w = self.width()
        f = QtGui.QFont(MONO)
        f.setPointSize(9)
        p.setFont(f)
        # ---- padded message bytes
        y = 24
        p.setPen(C_TEXT)
        fb = QtGui.QFont(p.font())
        fb.setBold(True)
        p.setFont(fb)
        extra = ""
        if run.extra_inputs:
            extra = f" + {len(run.extra_inputs)} extra input(s), each padded on its own (blue)"
        p.drawText(12, y, f"1. Padded input: {len(run.message)} message byte(s) + {len(run.padding)} padding byte(s)"
                          f"{extra} = {len(run.padded)} bytes = {len(run.padded) // v.rate_bytes} block(s) of r = {v.rate_bytes}")
        p.setFont(f)
        y += 10
        bw = 24
        per_row = max(8, (w - 40) // bw)
        rows_shown = 0
        max_bytes = per_row * 6
        data = run.padded
        for i, b in enumerate(data[:max_bytes]):
            r_, c_ = divmod(i, per_row)
            rect = QtCore.QRect(12 + c_ * bw, y + r_ * 20, bw - 2, 18)
            is_pad = run.is_padding(i)
            block = i // v.rate_bytes
            is_extra = run.segment_of(i) != "message"
            base = (C_PAD.darker(150) if is_pad else
                    QtGui.QColor(50, 80, 150) if is_extra else
                    (C_RATE.darker(150) if block % 2 == 0 else C_RATE.darker(120)))
            p.fillRect(rect, base)
            p.setPen(C_TEXT if not is_pad else QtGui.QColor(255, 200, 120))
            p.drawText(rect, QtCore.Qt.AlignCenter, f"{b:02x}")
            if i % v.rate_bytes == 0:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                p.drawLine(rect.left() - 1, rect.top(), rect.left() - 1, rect.bottom())
            rows_shown = r_ + 1
        y += rows_shown * 20 + 6
        if len(data) > max_bytes:
            p.setPen(C_DIM)
            p.drawText(12, y + 10, f"… {len(data) - max_bytes} more bytes not shown")
            y += 18
        p.setPen(C_DIM)
        pad_desc = (f"padding = domain byte 0x{v.domain_byte:02x} ({S.DOMAIN_NAMES.get(v.domain_byte, 'custom')}: suffix bits + first pad bit), "
                    f"zeros, then 0x80 (last pad bit)" if len(run.padding) > 1
                    else f"padding = a single byte 0x{run.padding[0]:02x} = domain 0x{v.domain_byte:02x} | 0x80 (both pad bits share the byte)")
        p.drawText(12, y + 12, pad_desc)
        y += 34
        # ---- rate / capacity bar to scale
        p.setPen(C_TEXT)
        p.setFont(fb)
        p.drawText(12, y, f"2. State = 200 bytes: rate r = {v.rate_bytes} B ({v.rate_bits} bits) | capacity c = {v.capacity_bytes} B "
                          f"({v.capacity_bits} bits) → claimed security c/2 = {v.security_bits} bits")
        p.setFont(f)
        y += 10
        bar_w = w - 40
        rw = bar_w * v.rate_bytes / 200
        p.fillRect(QtCore.QRectF(12, y, rw, 28), C_RATE)
        p.fillRect(QtCore.QRectF(12 + rw, y, bar_w - rw, 28), C_CAP)
        p.setPen(C_TEXT)
        p.drawText(QtCore.QRectF(12, y, rw, 28), QtCore.Qt.AlignCenter, f"rate {v.rate_bytes} B — blocks are XORed here, output is read here")
        p.drawText(QtCore.QRectF(12 + rw, y, bar_w - rw, 28), QtCore.Qt.AlignCenter, f"capacity {v.capacity_bytes} B")
        by_rate = {}
        for name, vv in S.VARIANTS.items():
            if name.startswith(("SHA3", "SHAKE")):
                by_rate.setdefault(vv.rate_bytes, []).append(name)
        for k, (rb, names) in enumerate(sorted(by_rate.items())):
            xx = 12 + bar_w * rb / 200
            p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 210, 120), 1, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(xx, y - 2), QtCore.QPointF(xx, y + 34 + 14 * (k % 2)))
            p.setPen(C_DIM if rb != v.rate_bytes else QtGui.QColor(255, 230, 90))
            label = "/".join(names) + f" r={rb}"
            p.drawText(QtCore.QPointF(xx - 40, y + 46 + 14 * (k % 2)), label)
        y += 76
        # ---- timeline
        p.setPen(C_TEXT)
        p.setFont(fb)
        n_perm = run.num_perm_calls
        p.drawText(12, y, f"3. Absorb / squeeze timeline — this {v.name} call costs {n_perm} permutation(s): "
                          f"{len(run.absorb_blocks)} absorb + {n_perm - len(run.absorb_blocks)} extra squeeze "
                          f"for {run.output_bytes} output byte(s) = ⌈{run.output_bytes}/{v.rate_bytes}⌉ = {max(1, -(-run.output_bytes // v.rate_bytes)) if run.output_bytes else 0} read(s)")
        p.setFont(f)
        y += 16
        self._perm_rects = []
        ev = run.timeline()
        box_w, gap = 92, 14
        per_row = max(1, (w - 40) // (box_w + gap))
        x = 12
        row = 0
        for k, (kind, idx) in enumerate(ev):
            if k and k % per_row == 0:
                row += 1
                x = 12
            yy = y + row * 74
            rect = QtCore.QRectF(x, yy, box_w, 54)
            if kind == "absorb":
                blk = run.absorb_blocks[idx]
                p.fillRect(rect, C_RATE.darker(130))
                p.setPen(C_TEXT)
                if blk.source != "message":
                    p.fillRect(rect, QtGui.QColor(50, 80, 150))
                p.drawText(rect.adjusted(4, 2, -2, -2), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                           f"XOR block {idx}\n{'into rate' if blk.source == 'message' else blk.source}\n{blk.data[:3].hex()}…")
            elif kind == "squeeze":
                sq = run.squeeze_blocks[idx]
                p.fillRect(rect, QtGui.QColor(70, 90, 140))
                p.setPen(C_TEXT)
                p.drawText(rect.adjusted(4, 2, -2, -2), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                           f"read {len(sq.data)} B\nfrom rate\n{sq.data[:3].hex()}…")
            else:
                cur = idx == s.perm_index
                p.fillRect(rect, QtGui.QColor(120, 80, 40) if cur else QtGui.QColor(90, 70, 50))
                if idx == self._hover_perm:
                    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
                    p.drawRect(rect)
                if cur:
                    p.setPen(QtGui.QPen(C_SEL, 2))
                    p.drawRect(rect)
                p.setPen(C_TEXT)
                p.drawText(rect.adjusted(4, 2, -2, -2), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                           f"f  #{idx + 1}\n{run.num_rounds} rounds\n(click → 3D)")
                self._perm_rects.append((rect, idx))
            # arrow
            if k + 1 < len(ev) and (k + 1) % per_row != 0:
                p.setPen(QtGui.QPen(C_DIM, 2))
                p.drawLine(QtCore.QPointF(x + box_w + 1, yy + 27), QtCore.QPointF(x + box_w + gap - 3, yy + 27))
                p.drawLine(QtCore.QPointF(x + box_w + gap - 7, yy + 22), QtCore.QPointF(x + box_w + gap - 3, yy + 27))
                p.drawLine(QtCore.QPointF(x + box_w + gap - 7, yy + 32), QtCore.QPointF(x + box_w + gap - 3, yy + 27))
            x += box_w + gap
        y += (row + 1) * 74 + 6
        # ---- output
        p.setPen(C_TEXT)
        p.setFont(fb)
        n_req = len(s.squeeze_requests())
        p.drawText(12, y, f"4. Output ({run.output_bytes} bytes = {run.output_bytes * 8} bits, "
                          f"{n_req} squeeze call(s)):")
        p.setFont(f)
        out = run.output.hex()
        y += 18
        p.setPen(QtGui.QColor(160, 230, 160))
        wtxt = f"as {s.params.word_bits}-bit words (Words-tab bit order): " + s.format_words(run.output, 0, 24)
        p.drawText(12, y, wtxt[:max(40, (w - 30) // 7)])
        y += 18
        chunk = max(32, (w - 40) // 8)
        for i in range(0, len(out), chunk):
            p.setPen(QtGui.QColor(160, 200, 255))
            p.drawText(12, y, out[i:i + chunk])
            y += 16
            if y > self.height() - 10:
                break
        self.setMinimumHeight(max(520, y + 10))


class SpongeView(QtWidgets.QWidget):
    """Message editor + the canvas."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(12, 8, 12, 4)
        top.addWidget(QtWidgets.QLabel("message:"))
        self.edit = QtWidgets.QLineEdit()
        self.edit.setFont(MONO)
        self.edit.setPlaceholderText("type text, or hex bytes with the checkbox on")
        self.edit.editingFinished.connect(self._apply)
        top.addWidget(self.edit, 1)
        self.hex_cb = QtWidgets.QCheckBox("hex")
        self.hex_cb.toggled.connect(self._toggle_hex)
        top.addWidget(self.hex_cb)
        self.len_label = QtWidgets.QLabel("")
        top.addWidget(self.len_label)
        self.counter = QtWidgets.QLabel("")
        self.counter.setStyleSheet("font-weight: bold; color: #ffd166;")
        top.addWidget(self.counter)
        lay.addLayout(top)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(12, 0, 12, 4)
        row.addWidget(QtWidgets.QLabel("more input after the permutation:"))
        self.extra_edit = QtWidgets.QLineEdit()
        self.extra_edit.setFont(MONO)
        self.extra_edit.setPlaceholderText("text, or hex with 0x prefix - absorbed as a new block after the last permutation")
        self.extra_edit.returnPressed.connect(self._add_extra)
        row.addWidget(self.extra_edit, 1)
        b = QtWidgets.QPushButton("absorb")
        b.clicked.connect(self._add_extra)
        row.addWidget(b)
        b = QtWidgets.QPushButton("remove extra inputs")
        b.clicked.connect(session.clear_inputs)
        row.addWidget(b)
        lay.addLayout(row)
        self.canvas = SpongeCanvas(session)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)
        lay.addWidget(scroll, 1)
        session.runChanged.connect(self._sync)
        self._sync()

    def activated(self) -> None:
        self._sync()

    def _sync(self) -> None:
        p = self.session.params
        self.hex_cb.blockSignals(True)
        self.hex_cb.setChecked(p.message_is_hex)
        self.hex_cb.blockSignals(False)
        txt = p.message.hex() if p.message_is_hex else p.message.decode("utf-8", "replace")
        if self.edit.text() != txt and not self.edit.hasFocus():
            self.edit.setText(txt)
        run = self.session.run
        self.len_label.setText(f"{len(p.message)} bytes")
        self.counter.setText(f"permutation calls: {run.num_perm_calls}")
        self.canvas.update()

    def _add_extra(self) -> None:
        txt = self.extra_edit.text().strip()
        try:
            data = bytes.fromhex(txt[2:].replace(" ", "")) if txt.lower().startswith("0x") else txt.encode("utf-8")
        except ValueError:
            self.len_label.setText("invalid hex")
            return
        self.extra_edit.clear()
        self.session.add_input(data)

    def _toggle_hex(self, on: bool) -> None:
        p = self.session.params
        self.edit.setText(p.message.hex() if on else p.message.decode("utf-8", "replace"))
        self.session.set_params(message_is_hex=on)

    def _apply(self) -> None:
        txt = self.edit.text()
        if self.hex_cb.isChecked():
            try:
                msg = bytes.fromhex(txt.replace(" ", ""))
            except ValueError:
                self.len_label.setText("invalid hex")
                return
        else:
            msg = txt.encode("utf-8")
        self.session.set_params(message=msg)
