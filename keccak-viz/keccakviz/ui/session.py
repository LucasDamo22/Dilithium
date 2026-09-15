"""The session: parameters -> sponge run -> current position in the trace.

Every view listens to this object.  It is the only place that owns state;
views are pure functions of (run, position, selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

from ..core import analysis as A
from ..core import keccak as K
from ..core import sponge as S

Cell = Tuple[int, int, int]

MAX_MESSAGE_BYTES = 16 * 1024


@dataclass
class Params:
    variant: str = "SHA3-256"
    domain_byte: int = S.DOMAIN_SHA3
    rate_bytes: int = 136
    num_rounds: int = 24
    output_bytes: int = 32
    message: bytes = b"abc"
    message_is_hex: bool = False
    enabled_steps: Tuple[str, ...] = K.STEP_NAMES
    round_offset_standard: bool = True  # reduced rounds use the last RCs (FIPS 202)
    avalanche_flip: Cell = (0, 0, 0)
    diffusion_source: Cell = (0, 0, 0)
    batch_n: int = 4

    def variant_obj(self) -> S.Variant:
        base = S.VARIANTS[self.variant]
        return base.with_(rate_bytes=self.rate_bytes, domain_byte=self.domain_byte)

    def copy(self) -> "Params":
        return Params(**self.__dict__)


class Session(QtCore.QObject):
    """Owns the parameters, the computed run, and the playback position."""

    runChanged = QtCore.pyqtSignal()  # parameters changed, run recomputed
    positionChanged = QtCore.pyqtSignal(int, int)  # perm index, snapshot index
    selectionChanged = QtCore.pyqtSignal(object)  # Cell or None
    structureChanged = QtCore.pyqtSignal(object)  # highlighted substructure name or None
    colorModeChanged = QtCore.pyqtSignal(str)
    detailLevelChanged = QtCore.pyqtSignal(int)
    jump_to_cube = QtCore.pyqtSignal()  # a view asks the main window to show the 3D cube

    STRUCTURES = ("row", "column", "lane", "slice", "plane", "sheet")
    COLOR_MODES = ("raw", "changed", "avalanche")

    def __init__(self, params: Optional[Params] = None):
        super().__init__()
        self.params = params or Params()
        self.run: S.SpongeRun = None  # type: ignore
        self.perm_index = 0
        self.snap_index = 0
        self.selected: Optional[Cell] = None
        self.structure: Optional[str] = None
        self.color_mode = "raw"
        self.detail_level = 1  # 0 plain, 1 student, 2 expert
        self._avalanche_cache: dict = {}
        self._diffusion_cache: dict = {}
        self.recompute()

    # ------------------------------------------------------------ parameters

    def set_params(self, **kw) -> None:
        changed = False
        for k, v in kw.items():
            if getattr(self.params, k) != v:
                setattr(self.params, k, v)
                changed = True
        if changed:
            self.recompute()

    def apply_variant(self, name: str) -> None:
        v = S.VARIANTS[name]
        self.params.variant = name
        self.params.rate_bytes = v.rate_bytes
        self.params.domain_byte = v.domain_byte
        if v.output_bytes is not None:
            self.params.output_bytes = v.output_bytes
        elif self.params.output_bytes == 0:
            self.params.output_bytes = 32
        self.recompute()

    def recompute(self) -> None:
        p = self.params
        msg = p.message[:MAX_MESSAGE_BYTES]
        self.run = S.sponge(
            msg,
            p.variant_obj(),
            p.output_bytes,
            p.num_rounds,
            p.enabled_steps,
            trace=True,
            round_offset=None if p.round_offset_standard else 0,
        )
        self._avalanche_cache.clear()
        self._diffusion_cache.clear()
        self.perm_index = min(self.perm_index, self.run.num_perm_calls - 1)
        self.snap_index = min(self.snap_index, self.num_snapshots - 1)
        self.runChanged.emit()
        self.positionChanged.emit(self.perm_index, self.snap_index)

    # ------------------------------------------------------------ position

    @property
    def perm(self) -> S.PermCall:
        return self.run.perm_calls[self.perm_index]

    @property
    def trace(self) -> K.Trace:
        return self.perm.trace

    @property
    def num_snapshots(self) -> int:
        return len(self.trace)

    @property
    def snapshot(self) -> K.Snapshot:
        return self.trace[self.snap_index]

    @property
    def prev_snapshot(self) -> K.Snapshot:
        return self.trace[max(0, self.snap_index - 1)]

    @property
    def state(self) -> np.ndarray:
        return self.snapshot.state

    def set_position(self, perm_index: Optional[int] = None, snap_index: Optional[int] = None) -> bool:
        pi = self.perm_index if perm_index is None else perm_index
        pi = max(0, min(pi, self.run.num_perm_calls - 1))
        n = len(self.run.perm_calls[pi].trace)
        si = self.snap_index if snap_index is None else snap_index
        si = max(0, min(si, n - 1))
        if (pi, si) == (self.perm_index, self.snap_index):
            return False
        self.perm_index, self.snap_index = pi, si
        self.positionChanged.emit(pi, si)
        return True

    def step_forward(self) -> bool:
        if self.snap_index + 1 < self.num_snapshots:
            return self.set_position(snap_index=self.snap_index + 1)
        if self.perm_index + 1 < self.run.num_perm_calls:
            return self.set_position(perm_index=self.perm_index + 1, snap_index=0)
        return False

    def step_back(self) -> bool:
        if self.snap_index > 0:
            return self.set_position(snap_index=self.snap_index - 1)
        if self.perm_index > 0:
            pi = self.perm_index - 1
            return self.set_position(perm_index=pi, snap_index=len(self.run.perm_calls[pi].trace) - 1)
        return False

    def round_forward(self) -> bool:
        # go to the end of the current round (or the next one if already there)
        ends = self.trace.round_end_indices()
        for e in ends:
            if e > self.snap_index:
                return self.set_position(snap_index=e)
        return self.set_position(snap_index=self.num_snapshots - 1)

    def round_back(self) -> bool:
        ends = [0] + self.trace.round_end_indices()
        for e in reversed(ends):
            if e < self.snap_index:
                return self.set_position(snap_index=e)
        return False

    def go_start(self) -> bool:
        return self.set_position(snap_index=0)

    def go_end(self) -> bool:
        return self.set_position(snap_index=self.num_snapshots - 1)

    # ------------------------------------------------------------ selection

    def select(self, cell: Optional[Cell]) -> None:
        if cell != self.selected:
            self.selected = cell
            self.selectionChanged.emit(cell)

    def set_structure(self, name: Optional[str]) -> None:
        if name != self.structure:
            self.structure = name
            self.structureChanged.emit(name)

    def set_color_mode(self, mode: str) -> None:
        if mode != self.color_mode:
            self.color_mode = mode
            self.colorModeChanged.emit(mode)

    def set_detail_level(self, level: int) -> None:
        if level != self.detail_level:
            self.detail_level = level
            self.detailLevelChanged.emit(level)

    # ------------------------------------------------------------ analyses

    def avalanche(self) -> A.Avalanche:
        key = (self.perm_index, self.params.avalanche_flip)
        if key not in self._avalanche_cache:
            p = self.params
            self._avalanche_cache[key] = A.avalanche(
                self.perm.state_in, [p.avalanche_flip], p.num_rounds, p.enabled_steps,
                None if p.round_offset_standard else 0, base_trace=self.trace)
        return self._avalanche_cache[key]

    def diffusion(self) -> A.Diffusion:
        key = (self.perm_index, self.params.diffusion_source)
        if key not in self._diffusion_cache:
            p = self.params
            self._diffusion_cache[key] = A.diffusion(
                self.perm.state_in, p.diffusion_source, p.num_rounds, p.enabled_steps,
                None if p.round_offset_standard else 0, base_trace=self.trace)
        return self._diffusion_cache[key]

    # ------------------------------------------------------------ describing

    def position_text(self) -> str:
        s = self.snapshot
        pc = self.perm
        head = f"permutation {pc.index + 1}/{self.run.num_perm_calls} ({pc.phase} block {pc.block_index})"
        if s.step == "initial":
            return f"{head} · initial state"
        skip = "  [disabled]" if s.skipped else ""
        return f"{head} · round {s.round + 1}/{self.params.num_rounds} · {K.STEP_SYMBOLS[s.step]} {s.step}{skip}"
