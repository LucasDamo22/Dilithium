"""Step / round / play controls and the position scrubber."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..core import keccak as K
from .session import Session


class Transport(QtWidgets.QWidget):
    """Buttons + scrubber; drives ``session`` and owns the play timer."""

    speedChanged = QtCore.pyqtSignal(int)  # animation duration in ms
    scrubbed = QtCore.pyqtSignal(float)  # user dragged the step scrubber to t in [0, 1]

    ZONE = 0.05  # fraction at each end that hands over to the neighbouring step
    DEAD = 0.08  # dead zone before it, where the handle sticks at t = 0 / t = 1

    SPEEDS = [(0, "instant"), (250, "fast"), (700, "normal"), (1400, "slow"), (3000, "very slow")]

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_play_tick)
        self._anim_ms = 700

        from .flow import FlowLayout

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.setSpacing(2)
        lay = FlowLayout()
        row1 = QtWidgets.QWidget()
        row1.setLayout(lay)
        outer.addWidget(row1)

        def btn(text, tip, slot):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            b.setAutoRaise(True)
            lay.addWidget(b)
            return b

        btn("⏮", "Initial state (Home)", session.go_start)
        btn("⏪", "Previous round (Down / PgUp)", session.round_back)
        btn("◀", "Previous step (Left)", session.step_back)
        self.play_btn = btn("▶ play", "Play / pause (Space or P)", self.toggle_play)
        btn("▶", "Next step (Right)", session.step_forward)
        btn("⏩", "Next round (Up / PgDown)", session.round_forward)
        btn("⏭", "Final state (End)", session.go_end)

        lay.addSpacing(10)
        self.perm_prev = btn("◁ perm", "Previous permutation call of the sponge run", self._perm_prev)
        self.perm_combo = QtWidgets.QComboBox()
        self.perm_combo.setToolTip("Which permutation call of the sponge run.  A message longer than the rate, "
                                   "or SHAKE output longer than the rate, needs more than one call.")
        self.perm_combo.currentIndexChanged.connect(self._on_perm_combo)
        lay.addWidget(self.perm_combo)
        self.perm_next = btn("perm ▷", "Next permutation call of the sponge run", self._perm_next)
        self.add_btn = btn("+ input", "Absorb more data after the last permutation (duplex-style: the extra input is "
                                      "padded on its own, then gets its own 24 rounds)", self._add_input)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setToolTip("Scrub through every step mapping of this permutation")
        self.slider.setMinimumWidth(160)
        self.slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self.slider)

        self.pos_label = QtWidgets.QLabel()
        self.pos_label.setMinimumWidth(150)
        lay.addWidget(self.pos_label)

        lay.addWidget(QtWidgets.QLabel("speed"))
        self.speed = QtWidgets.QComboBox()
        for ms, name in self.SPEEDS:
            self.speed.addItem(name, ms)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._on_speed)
        lay.addWidget(self.speed)

        # ---- step scrubber: drags the current step's animation; the end zones hand over
        row2 = QtWidgets.QHBoxLayout()
        outer.addLayout(row2)
        self.step_label = QtWidgets.QLabel("step animation")
        row2.addWidget(self.step_label)
        self.step_slider = ZoneSlider(self.ZONE, self.DEAD)
        self.step_slider.setRange(0, 1000)
        self.step_slider.setValue(1000)
        self.step_slider.setMinimumWidth(200)
        self.step_slider.setToolTip("Drag to play the current step mapping by hand.  Near each end the handle sticks "
                                    "at the finished / initial state; push all the way into the end zone to take the "
                                    "next step (right) or go back one step (left).")
        self.step_slider.sliderPressed.connect(self._scrub_start)
        self.step_slider.sliderReleased.connect(self._scrub_end)
        self.step_slider.valueChanged.connect(self._scrub_value)
        row2.addWidget(self.step_slider, 1)
        self.t_label = QtWidgets.QLabel("t = 1.00")
        self.t_label.setMinimumWidth(70)
        row2.addWidget(self.t_label)
        self._scrubbing = False
        self._armed = True
        self._last_v = 1000

        session.runChanged.connect(self._on_run)
        session.positionChanged.connect(self._on_position)
        self._on_run()

    # ------------------------------------------------------------

    @property
    def anim_ms(self) -> int:
        return self._anim_ms

    # ---- scrubber

    def set_progress(self, t: float) -> None:
        """Called by the animator while it plays; ignored while the user drags."""
        if self._scrubbing:
            return
        lo = int((self.ZONE + self.DEAD) * 1000)
        hi = 1000 - lo
        self._set_slider(int(round(lo + t * (hi - lo))))
        self.t_label.setText(f"t = {t:4.2f}")

    def _scrub_start(self) -> None:
        self.stop()
        self._scrubbing = True
        self._armed = True
        self._last_v = self.step_slider.value()

    def _scrub_end(self) -> None:
        self._scrubbing = False
        self._armed = True
        self.step_slider.locked = False

    def _scrub_value(self, v: int) -> None:
        if not self._scrubbing or self.step_slider.locked:
            return
        s = self.session
        zone = int(self.ZONE * 1000)
        dead_hi = 1000 - int((self.ZONE + self.DEAD) * 1000)  # start of the upper dead zone
        dead_lo = int((self.ZONE + self.DEAD) * 1000)  # end of the lower dead zone
        if self._armed and v >= 1000 - zone and s.snap_index + 1 < s.num_snapshots:
            # hand over: finish this step, take the next one, hold it at t = 0 until release
            self._armed = False
            s.step_forward()
            self._set_slider(0)
            self.step_slider.locked = True
            self.scrubbed.emit(0.0)
        elif self._armed and v <= zone and s.snap_index > 0:
            # hand over: back to the previous step, shown complete, held until release
            self._armed = False
            s.step_back()
            self._set_slider(1000)
            self.step_slider.locked = True
            self.scrubbed.emit(1.0)
        elif v >= dead_hi:
            # dead zone: stick at the finished state so the result can be seen
            self._set_slider(dead_hi)
            self.scrubbed.emit(1.0)
        elif v <= dead_lo:
            self._set_slider(dead_lo)
            self.scrubbed.emit(0.0)
        else:
            t = (v - dead_lo) / float(dead_hi - dead_lo)
            self.t_label.setText(f"t = {t:4.2f}")
            self.scrubbed.emit(t)
        self._last_v = self.step_slider.value()

    def _set_slider(self, v: int) -> None:
        self.step_slider.blockSignals(True)
        self.step_slider.setValue(v)
        self.step_slider.blockSignals(False)
        self._last_v = v
        lo = int((self.ZONE + self.DEAD) * 1000)
        t = max(0.0, min(1.0, (v - lo) / float(1000 - 2 * lo)))
        self.t_label.setText(f"t = {t:4.2f}")

    def _add_input(self) -> None:
        self.stop()
        txt, ok = QtWidgets.QInputDialog.getText(
            self, "More input", "Data to absorb after the last permutation\n(text, or hex with a 0x prefix):",
            text="more input")
        if not ok:
            return
        txt = txt.strip()
        try:
            data = bytes.fromhex(txt[2:].replace(" ", "")) if txt.lower().startswith("0x") else txt.encode("utf-8")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "More input", "Not valid hex.")
            return
        self.session.add_input(data)

    def _perm_prev(self) -> None:
        self.session.set_position(perm_index=self.session.perm_index - 1, snap_index=0)

    def _perm_next(self) -> None:
        self.session.set_position(perm_index=self.session.perm_index + 1, snap_index=0)

    def _on_speed(self, _i: int) -> None:
        self._anim_ms = int(self.speed.currentData())
        self.speedChanged.emit(self._anim_ms)
        if self._timer.isActive():
            self._timer.setInterval(self._play_interval())

    def _play_interval(self) -> int:
        return max(60, self._anim_ms + 120)

    def toggle_play(self) -> None:
        if self._timer.isActive():
            self.stop()
        else:
            if self.session.snap_index >= self.session.num_snapshots - 1 \
                    and self.session.perm_index >= self.session.run.num_perm_calls - 1:
                self.session.set_position(perm_index=0, snap_index=0)
            self._timer.start(self._play_interval())
            self.play_btn.setText("⏸ pause")

    def stop(self) -> None:
        self._timer.stop()
        self.play_btn.setText("▶ play")

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def _on_play_tick(self) -> None:
        s = self.session
        if s.snap_index + 1 >= s.num_snapshots:
            # end of this permutation call: stop here rather than running into the next call
            self.stop()
            return
        if not s.step_forward():
            self.stop()

    def _on_run(self) -> None:
        s = self.session
        self.perm_combo.blockSignals(True)
        self.perm_combo.clear()
        n = s.run.num_perm_calls
        for c in s.run.perm_calls:
            self.perm_combo.addItem(f"permutation {c.index + 1} of {n} ({c.phase} block {c.block_index})")
        self.perm_combo.setCurrentIndex(s.perm_index)
        self.perm_combo.blockSignals(False)
        self.perm_prev.setEnabled(n > 1)
        self.perm_next.setEnabled(n > 1)
        self._sync_slider()

    def _sync_slider(self) -> None:
        s = self.session
        self.slider.blockSignals(True)
        self.slider.setRange(0, s.num_snapshots - 1)
        self.slider.setValue(s.snap_index)
        self.slider.blockSignals(False)
        snap = s.snapshot
        if snap.step == "initial":
            self.pos_label.setText("initial state")
            self.step_label.setText("step animation (none yet)")
        elif snap.step == "squeeze":
            info = snap.info or {}
            self.pos_label.setText(f"squeeze: {info.get('nbits')} output bits")
            self.step_label.setText("animate the read-out by hand")
        elif snap.step == "load":
            info = snap.info or {}
            phase = info.get("phase")
            self.pos_label.setText({"seed": "block on the bus", "iv": "state register"}.get(
                phase, f"loading cycle {info.get('cycle')}/{info.get('n_cycles')}"))
            self.step_label.setText("animate the bus cycle by hand")
        else:
            self.pos_label.setText(f"round {snap.round + 1}/{s.params.num_rounds}  ·  "
                                   f"{K.STEP_SYMBOLS[snap.step]} {snap.step}")
            self.step_label.setText(f"animate {K.STEP_SYMBOLS[snap.step]} {snap.step} by hand")
        self.perm_prev.setEnabled(s.perm_index > 0)
        self.perm_next.setEnabled(s.perm_index + 1 < s.run.num_perm_calls)

    def _on_position(self, perm: int, snap: int) -> None:
        if self.perm_combo.currentIndex() != perm:
            self.perm_combo.blockSignals(True)
            self.perm_combo.setCurrentIndex(perm)
            self.perm_combo.blockSignals(False)
        self._sync_slider()

    def _on_slider(self, v: int) -> None:
        self.session.set_position(snap_index=v)

    def _on_perm_combo(self, i: int) -> None:
        if i >= 0:
            self.session.set_position(perm_index=i, snap_index=0)


class ZoneSlider(QtWidgets.QSlider):
    """A horizontal slider that paints its hand-over and dead zones and can be
    locked (ignores the mouse until release, used right after a hand-over)."""

    def __init__(self, zone: float, dead: float, parent=None):
        super().__init__(QtCore.Qt.Horizontal, parent)
        self.zone = zone
        self.dead = dead
        self.locked = False
        self.setFixedHeight(34)

    def mousePressEvent(self, ev):
        # jump the handle to the click position so a drag can start anywhere
        if ev.button() == QtCore.Qt.LeftButton:
            v = QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), ev.x(), self.width())
            self.setSliderDown(True)
            self.sliderPressed.emit()
            self.setValue(v)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.isSliderDown():
            self.setSliderDown(False)
            self.sliderReleased.emit()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.isSliderDown():
            if not self.locked:
                v = QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), ev.x(), self.width())
                self.setValue(v)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def paintEvent(self, ev):
        from PyQt5 import QtGui
        p = QtGui.QPainter(self)
        w = self.width()
        zw = int(w * self.zone)
        dw = int(w * self.dead)
        gy = 4  # groove band at the top; labels go underneath
        p.fillRect(0, gy, zw, 12, QtGui.QColor(110, 80, 170, 160))
        p.fillRect(zw, gy, dw, 12, QtGui.QColor(70, 70, 90, 140))
        p.fillRect(w - zw - dw, gy, dw, 12, QtGui.QColor(70, 70, 90, 140))
        p.fillRect(w - zw, gy, zw, 12, QtGui.QColor(70, 150, 100, 160))
        p.setPen(QtGui.QColor(170, 170, 185))
        f = p.font()
        f.setPointSize(7)
        p.setFont(f)
        p.drawText(QtCore.QRect(0, 18, zw + dw, 14), QtCore.Qt.AlignLeft, "◀ prev step  | hold t=0")
        p.drawText(QtCore.QRect(w - zw - dw, 18, zw + dw, 14), QtCore.Qt.AlignRight, "hold t=1 |  next step ▶")
        p.end()
        super().paintEvent(ev)
