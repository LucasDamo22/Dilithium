"""Step / round / play controls and the position scrubber."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..core import keccak as K
from .session import Session


class Transport(QtWidgets.QWidget):
    """Buttons + scrubber; drives ``session`` and owns the play timer."""

    speedChanged = QtCore.pyqtSignal(int)  # animation duration in ms

    SPEEDS = [(0, "instant"), (250, "fast"), (700, "normal"), (1400, "slow"), (3000, "very slow")]

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_play_tick)
        self._anim_ms = 700

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)

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

        self.perm_combo = QtWidgets.QComboBox()
        self.perm_combo.setToolTip("Which permutation call of the sponge run")
        self.perm_combo.currentIndexChanged.connect(self._on_perm_combo)
        lay.addWidget(self.perm_combo)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setToolTip("Scrub through every step mapping of this permutation")
        self.slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self.slider, 1)

        self.pos_label = QtWidgets.QLabel()
        self.pos_label.setMinimumWidth(190)
        lay.addWidget(self.pos_label)

        lay.addWidget(QtWidgets.QLabel("speed"))
        self.speed = QtWidgets.QComboBox()
        for ms, name in self.SPEEDS:
            self.speed.addItem(name, ms)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._on_speed)
        lay.addWidget(self.speed)

        session.runChanged.connect(self._on_run)
        session.positionChanged.connect(self._on_position)
        self._on_run()

    # ------------------------------------------------------------

    @property
    def anim_ms(self) -> int:
        return self._anim_ms

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
        if not self.session.step_forward():
            self.stop()

    def _on_run(self) -> None:
        s = self.session
        self.perm_combo.blockSignals(True)
        self.perm_combo.clear()
        for c in s.run.perm_calls:
            self.perm_combo.addItem(f"perm {c.index + 1}/{s.run.num_perm_calls} ({c.phase} #{c.block_index})")
        self.perm_combo.setCurrentIndex(s.perm_index)
        self.perm_combo.blockSignals(False)
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
        else:
            self.pos_label.setText(f"round {snap.round + 1}/{s.params.num_rounds}  ·  "
                                   f"{K.STEP_SYMBOLS[snap.step]} {snap.step}")

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
