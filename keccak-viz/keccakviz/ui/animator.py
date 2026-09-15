"""One shared clock for step animations.

Every view that animates (the 3D cube, the slice stack) reads the same
``StepAnimator`` so that playing, stepping and scrubbing stay in sync.
"""

from __future__ import annotations

import time
from typing import Tuple

from PyQt5 import QtCore

from ..core import keccak as K
from .session import Session


class StepAnimator(QtCore.QObject):
    progress = QtCore.pyqtSignal(float)  # t of the step being shown
    changed = QtCore.pyqtSignal()  # views should repaint

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.animation_ms = 700
        self.target = session.snap_index  # snapshot whose step is (being) shown
        self.t = 1.0
        self._dir = 1
        self._start = 0.0
        self._frozen = False
        self._last = (session.perm_index, session.snap_index)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        session.positionChanged.connect(self._on_position)
        session.runChanged.connect(self._on_run)

    # ------------------------------------------------------------ state

    @property
    def active(self) -> bool:
        """True while a step is shown part-way (timer running or frozen by the scrubber)."""
        return self._timer.isActive() or self._frozen

    def in_transition(self) -> bool:
        s = self.session
        if not self.active or self.target >= s.num_snapshots or self.t >= 1.0:
            return False
        return s.trace[self.target].step != "initial"

    def pair(self) -> Tuple[K.Snapshot, K.Snapshot, float]:
        """(previous snapshot, snapshot being shown, t)."""
        s = self.session
        tr = s.trace
        if self.active and self.target < len(tr):
            snap = tr[self.target]
            return tr[max(0, self.target - 1)], snap, self.t
        return s.prev_snapshot, s.snapshot, 1.0

    # ------------------------------------------------------------ control

    def set_animation_ms(self, ms: int) -> None:
        self.animation_ms = ms

    def freeze(self, t: float) -> None:
        """Hold the current snapshot's step at time t (scrubbing, screenshots)."""
        self._timer.stop()
        self.target = self.session.snap_index
        self.t = max(0.0, min(1.0, float(t)))
        self._frozen = True
        self.progress.emit(self.t)
        self.changed.emit()

    def _on_run(self) -> None:
        self._last = (self.session.perm_index, self.session.snap_index)
        self._timer.stop()
        self._frozen = False
        self.t = 1.0
        self.target = self.session.snap_index
        self.progress.emit(1.0)
        self.changed.emit()

    def _on_position(self, perm: int, snap: int) -> None:
        self._frozen = False
        lp, ls = self._last
        if perm == lp and snap == ls + 1:
            self._start_anim(snap, +1)
        elif perm == lp and snap == ls - 1:
            self._start_anim(ls, -1)
        else:
            self._timer.stop()
            self.t = 1.0
            self.target = snap
            self.progress.emit(1.0)
        self._last = (perm, snap)
        self.changed.emit()

    def _start_anim(self, step_snap: int, direction: int) -> None:
        self.target = step_snap
        self._dir = direction
        self.t = 0.0 if direction > 0 else 1.0
        self._start = time.perf_counter()
        if self.animation_ms <= 0:
            self.t = 1.0
            self.progress.emit(1.0)
            return
        self.progress.emit(self.t)
        self._timer.start()

    def _tick(self) -> None:
        el = (time.perf_counter() - self._start) * 1000.0
        u = min(1.0, el / max(1, self.animation_ms))
        self.t = u if self._dir > 0 else 1.0 - u
        if u >= 1.0:
            self._timer.stop()
        self.progress.emit(self.t)
        self.changed.emit()
