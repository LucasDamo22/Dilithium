"""The session: parameters -> sponge run -> current position in the trace.

Every view listens to this object.  It is the only place that owns state;
views are pure functions of (run, position, selection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    word_bits: int = 64  # word size for the per-word view and the loading bus
    words_per_cycle: int = 1  # bus width = word_bits * words_per_cycle bits per load cycle
    show_load: bool = True  # prepend the loading phase to each absorb call

    def variant_obj(self) -> S.Variant:
        base = S.VARIANTS[self.variant]
        return base.with_(rate_bytes=self.rate_bytes, domain_byte=self.domain_byte)


@dataclass
class Dye:
    """A colour attached to a set of bits; it spreads along the dependency pattern."""

    label: str
    origins: List[Cell]
    color: str  # "#rrggbb"


@dataclass
class TrackGroup:
    """A set of bits followed together under one colour."""

    label: str
    origins: List[Cell]  # positions in snapshot 0

    @property
    def single(self) -> bool:
        return len(self.origins) == 1


class Session(QtCore.QObject):
    """Owns the parameters, the computed run, and the playback position."""

    runChanged = QtCore.pyqtSignal()  # parameters changed, run recomputed
    positionChanged = QtCore.pyqtSignal(int, int)  # perm index, snapshot index
    selectionChanged = QtCore.pyqtSignal(object)  # Cell or None
    structureChanged = QtCore.pyqtSignal(object)  # highlighted substructure name or None
    colorModeChanged = QtCore.pyqtSignal(str)
    detailLevelChanged = QtCore.pyqtSignal(int)
    jump_to_cube = QtCore.pyqtSignal()  # a view asks the main window to show the 3D cube
    trackedChanged = QtCore.pyqtSignal()  # the list of tracked bits changed
    styleChanged = QtCore.pyqtSignal(str)  # cell representation changed
    visibilityChanged = QtCore.pyqtSignal(str)  # both / ones / zeros
    wordsChanged = QtCore.pyqtSignal(bool)  # show word grouping on the cube / slices
    dyesChanged = QtCore.pyqtSignal()
    pulledChanged = QtCore.pyqtSignal()

    STRUCTURES = ("row", "column", "lane", "slice", "plane", "sheet")
    CELL_STYLES = (
        ("cubes", "big cube = 1, small dark cube = 0"),
        ("equal", "equal cubes: orange = 1, blue = 0"),
        ("spheres", "spheres: big = 1, dot = 0"),
        ("mono", "same size, white = 1, black = 0"),
    )
    VISIBILITY = (("both", "show 0 and 1"), ("ones", "only the 1 bits"), ("zeros", "only the 0 bits"))
    COLOR_MODES = ("raw", "changed", "avalanche", "dye")
    DYE_COLORS = ("#ff3b3b", "#3b8bff", "#3bff6a", "#ffd23b", "#ff3bd6", "#3bf0ff", "#ff8c3b", "#c03bff")
    MAX_DYES = 8
    WORD_SIZES = (1, 2, 4, 8, 16, 32, 64)
    TRACK_COLORS = ("#4cff7a", "#4cd7ff", "#ff9f4c", "#ff4cf0", "#f5ff4c", "#ffffff", "#b58cff", "#ff6b6b")
    MAX_TRACKED = len(TRACK_COLORS)

    def __init__(self, params: Optional[Params] = None):
        super().__init__()
        self.params = params or Params()
        self.run: S.SpongeRun = None  # type: ignore
        self.perm_index = 0
        self.snap_index = 0
        self.selected: Optional[Cell] = None
        self.structure: Optional[str] = None
        self.color_mode = "raw"
        self.cell_style = "cubes"
        self.visibility = "both"
        self.show_words = False
        self.detail_level = 1  # 0 plain, 1 student, 2 expert
        self._avalanche_cache: dict = {}
        self._diffusion_cache: dict = {}
        self.tracked: List[TrackGroup] = []
        self.dyes: List[Dye] = []
        self._dye_cache: dict = {}
        self.pulled: List[Tuple[str, Cell]] = []  # (structure name, anchor cell) regions lifted out of the cube
        self.pull_vectors: List[np.ndarray] = []  # world offset of each pulled region (draggable)
        self._trace_cache: dict = {}
        self._track_cache: Dict[Tuple[int, Cell], A.BitTrack] = {}
        self._bits_cache: Dict[Tuple[int, int], np.ndarray] = {}
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
        self._track_cache.clear()
        self._bits_cache.clear()
        self._dye_cache.clear()
        self._trace_cache.clear()
        self.perm_index = min(self.perm_index, self.run.num_perm_calls - 1)
        self.snap_index = min(self.snap_index, self.num_snapshots - 1)
        self.runChanged.emit()
        self.positionChanged.emit(self.perm_index, self.snap_index)

    # ------------------------------------------------------------ position

    @property
    def perm(self) -> S.PermCall:
        return self.run.perm_calls[self.perm_index]

    @property
    def trace(self) -> S.FullTrace:
        """The current permutation call's trace with its loading phase in front."""
        return self.trace_for(self.perm_index)

    def trace_for(self, perm_index: int) -> S.FullTrace:
        p = self.params
        key = (perm_index, p.word_bits, p.words_per_cycle, p.show_load)
        t = self._trace_cache.get(key)
        if t is None:
            call = self.run.perm_calls[perm_index]
            block = None
            if p.show_load and call.phase == "absorb":
                block = self.run.absorb_blocks[call.block_index]
            t = self._trace_cache[key] = S.full_trace(call, block, p.word_bits, p.words_per_cycle)
        return t

    def core_index(self, index: int) -> int:
        """Snapshot index in the bare permutation trace (loading frames map to 0)."""
        return self.trace.core_index(index)

    @property
    def n_load(self) -> int:
        return self.trace.n_load

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
        n = len(self.trace_for(pi))
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
            return self.set_position(perm_index=pi, snap_index=len(self.trace_for(pi)) - 1)
        return False

    def round_forward(self) -> bool:
        # go to the end of the current round (or the next one if already there)
        ends = self.trace.round_end_indices()
        for e in ends:
            if e > self.snap_index:
                return self.set_position(snap_index=e)
        return self.set_position(snap_index=self.num_snapshots - 1)

    def round_back(self) -> bool:
        ends = sorted({0, self.n_load} | set(self.trace.round_end_indices()))
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

    def set_cell_style(self, style: str) -> None:
        if style != self.cell_style:
            self.cell_style = style
            self.styleChanged.emit(style)

    # ------------------------------------------------------------ dyes

    def add_dye(self, cell: Optional[Cell], color: Optional[str] = None) -> bool:
        """Dye the focused bit (or the whole active substructure) with ``color``."""
        if cell is None or len(self.dyes) >= self.MAX_DYES:
            return False
        if self.structure:
            from . import anim

            mask = anim.structure_cells(self.structure, *cell)
            cells = [(int(anim.XS[i]), int(anim.YS[i]), int(anim.ZS[i])) for i in np.nonzero(mask)[0]]
            label = f"{self.structure} through ({cell[0]},{cell[1]},{cell[2]})"
        else:
            cells = [tuple(int(v) for v in cell)]
            label = f"bit ({cell[0]},{cell[1]},{cell[2]})"
        origins = [self.origin_of(c) for c in cells]
        color = color or self.DYE_COLORS[len(self.dyes) % len(self.DYE_COLORS)]
        self.dyes.append(Dye(label, origins, color))
        self._dye_cache.clear()
        self.dyesChanged.emit()
        if self.color_mode != "dye":
            self.set_color_mode("dye")
        return True

    def set_dye_color(self, index: int, color: str) -> None:
        if 0 <= index < len(self.dyes):
            self.dyes[index].color = color
            self.dyesChanged.emit()

    def remove_dye(self, index: int) -> None:
        if 0 <= index < len(self.dyes):
            del self.dyes[index]
            self._dye_cache.clear()
            self.dyesChanged.emit()

    def clear_dyes(self) -> None:
        if self.dyes:
            self.dyes.clear()
            self._dye_cache.clear()
            self.dyesChanged.emit()

    def dye_array(self) -> Optional[np.ndarray]:
        """(n_snapshots, 1600, K) dye concentrations for the current call, or None."""
        if not self.dyes:
            return None
        key = (self.perm_index, len(self.dyes), tuple(tuple(d.origins) for d in self.dyes),
               self.params.word_bits, self.params.words_per_cycle, self.params.show_load)
        arr = self._dye_cache.get(key)
        if arr is None:
            from . import anim

            init = np.zeros((1600, len(self.dyes)), dtype=np.float32)
            for k, d in enumerate(self.dyes):
                for (x, y, z) in d.origins:
                    init[anim.cell_index(x, y, z), k] = 1.0
            arr = self._dye_cache[key] = A.propagate_dye(self.trace, init)
        return arr

    def dye_render(self, index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(rgb (1600,3), strength (1600,)) of the dye mixture at snapshot ``index``.

        Hue is the concentration-weighted mix of the dye colours; strength is the
        cell's total concentration relative to the strongest cell right now, so
        the picture stays visible as the dye thins out and reads as uniform once
        it has spread evenly."""
        arr = self.dye_array()
        if arr is None:
            return None
        d = arr[min(index, len(arr) - 1)]
        tot = d.sum(axis=1)
        cols = np.array([[int(c.color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for c in self.dyes], dtype=np.float32)
        mix = (d @ cols) / np.maximum(tot, 1e-12)[:, None]
        strength = np.sqrt(tot / max(float(tot.max()), 1e-12))
        return mix.astype(np.float32), strength.astype(np.float32)

    def dye_render_groups(self, index: int, groups: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Like :meth:`dye_render` but for groups of cells (e.g. words): ``groups`` is an
        (n, m) array of cell indices; each group's dye is the sum over its cells, and
        strength is relative to the most-dyed group."""
        arr = self.dye_array()
        if arr is None:
            return None
        d = arr[min(index, len(arr) - 1)][groups].sum(axis=1)  # (n, K)
        tot = d.sum(axis=1)
        cols = np.array([[int(c.color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for c in self.dyes], dtype=np.float32)
        mix = (d @ cols) / np.maximum(tot, 1e-12)[:, None]
        strength = np.sqrt(tot / max(float(tot.max()), 1e-12))
        return mix.astype(np.float32), strength.astype(np.float32)

    def dye_spread(self, index: Optional[int] = None) -> Optional[float]:
        arr = self.dye_array()
        if arr is None:
            return None
        k = self.snap_index if index is None else index
        return A.dye_spread(arr[min(k, len(arr) - 1)])

    # ------------------------------------------------------------ pulled-out regions

    def pull_out(self, name: Optional[str], cell: Optional[Cell]) -> bool:
        """Lift the named substructure through ``cell`` (a region of positions) out of the cube."""
        if name is None or cell is None:
            return False
        entry = (name, tuple(int(v) for v in cell))
        if entry in self.pulled or len(self.pulled) >= 4:
            return False
        self.pulled.append(entry)
        self.pull_vectors.append(np.array([0.0, 8.0 + 7.0 * (len(self.pulled) - 1), 0.0], dtype=np.float32))
        self.pulledChanged.emit()
        return True

    def move_pulled(self, index: int, delta: np.ndarray) -> None:
        """Drag a pulled-out region by a world-space delta."""
        if 0 <= index < len(self.pulled):
            self.pull_vectors[index] = (self.pull_vectors[index] + np.asarray(delta, dtype=np.float32))
            self.pulledChanged.emit()

    def push_back(self, index: Optional[int] = None) -> None:
        if index is None:
            self.pulled.clear()
            self.pull_vectors.clear()
        elif 0 <= index < len(self.pulled):
            del self.pulled[index]
            del self.pull_vectors[index]
        self.pulledChanged.emit()

    def pull_offsets(self) -> Optional[np.ndarray]:
        """(1600, 3) world offset per cell position, or None when nothing is pulled out."""
        if not self.pulled:
            return None
        from . import anim

        off = np.zeros((1600, 3), dtype=np.float32)
        taken = np.zeros(1600, dtype=bool)
        for k, (name, cell) in enumerate(self.pulled):
            m = anim.structure_cells(name, *cell) & ~taken
            off[m] = self.pull_vectors[k]
            taken |= m
        return off

    def set_show_words(self, on: bool) -> None:
        if bool(on) != self.show_words:
            self.show_words = bool(on)
            self.wordsChanged.emit(self.show_words)

    def word_of(self, cell: Cell) -> Tuple[int, int, int]:
        """(word index, first bit, last bit) of the word containing ``cell``."""
        wb = max(1, self.params.word_bits)
        w = K.bit_index(*cell) // wb
        return w, w * wb, (w + 1) * wb - 1

    def set_visibility(self, vis: str) -> None:
        if vis != self.visibility:
            self.visibility = vis
            self.visibilityChanged.emit(vis)

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
                None if p.round_offset_standard else 0, base_trace=self.perm.trace)
        return self._avalanche_cache[key]

    def diffusion(self) -> A.Diffusion:
        key = (self.perm_index, self.params.diffusion_source)
        if key not in self._diffusion_cache:
            p = self.params
            self._diffusion_cache[key] = A.diffusion(
                self.perm.state_in, p.diffusion_source, p.num_rounds, p.enabled_steps,
                None if p.round_offset_standard else 0, base_trace=self.perm.trace)
        return self._diffusion_cache[key]

    # ------------------------------------------------------------ tracked bits

    def track(self, cell: Optional[Cell]) -> bool:
        """Start following the bit that sits at ``cell`` in the *current* snapshot.

        The origin stored is the cell's position in snapshot 0 (positions are
        data-independent, so the same origin is valid in every permutation call)."""
        if cell is None:
            return False
        x, y, z = cell
        return self._add_group(f"bit ({x},{y},{z})", [cell])

    def track_structure(self, name: str, cell: Optional[Cell]) -> bool:
        """Follow every bit of the named substructure (row, column, lane, slice,
        plane, sheet) through ``cell`` at the current snapshot, as one group."""
        if cell is None or name is None:
            return False
        from . import anim  # local import: anim has no Qt dependency but lives in ui

        mask = anim.structure_cells(name, *cell)
        cells = [(int(anim.XS[i]), int(anim.YS[i]), int(anim.ZS[i])) for i in np.nonzero(mask)[0]]
        x, y, z = cell
        return self._add_group(f"{name} through ({x},{y},{z})", cells)

    def track_focus(self, cell: Optional[Cell]) -> bool:
        """What the F key does: the highlighted structure if one is active, else the single bit."""
        if self.structure:
            return self.track_structure(self.structure, cell)
        return self.track(cell)

    def _add_group(self, label: str, cells) -> bool:
        if len(self.tracked) >= self.MAX_TRACKED:
            return False
        origins = [self.origin_of(c) for c in cells]
        if any(g.origins == origins for g in self.tracked):
            return False
        self.tracked.append(TrackGroup(label, origins))
        self.trackedChanged.emit()
        return True

    def origin_of(self, cell: Cell) -> Cell:
        """Position in snapshot 0 of the bit currently at ``cell``."""
        return self.origin_of_at(cell, self.snap_index)

    def origin_of_at(self, cell: Cell, index: int) -> Cell:
        """Position in snapshot 0 of the bit that sits at ``cell`` in snapshot ``index``."""
        pos = tuple(int(v) for v in cell)
        for snap in reversed(self.trace.snapshots[1:index + 1]):
            if snap.skipped:
                continue
            if snap.step == "rho":
                x, y, z = pos
                pos = (x, y, (z - int(K.RHO_OFFSETS[x, y])) % 64)
            elif snap.step == "pi":
                x, y, z = pos
                sx, sy = K.pi_source(x, y)
                pos = (sx, sy, z)
        return pos

    def untrack(self, index: int) -> None:
        if 0 <= index < len(self.tracked):
            del self.tracked[index]
            self.trackedChanged.emit()

    def clear_tracked(self) -> None:
        if self.tracked:
            self.tracked.clear()
            self.trackedChanged.emit()

    def snapshot_bits(self, index: int) -> np.ndarray:
        """Cached (5,5,64) bit array of snapshot ``index`` of the current permutation call."""
        key = (self.perm_index, index)
        b = self._bits_cache.get(key)
        if b is None:
            b = self._bits_cache[key] = K.lanes_to_bits(self.trace[index].state)
        return b

    def bit_track(self, origin: Cell) -> A.BitTrack:
        key = (self.perm_index, origin)
        if key not in self._track_cache:
            self._track_cache[key] = A.track_bit(self.trace, origin, self.snapshot_bits)
        return self._track_cache[key]

    def tracked_tracks(self) -> List[Tuple[int, Cell, A.BitTrack]]:
        """(colour index, origin, track) for every tracked bit in the current permutation call."""
        return [(i, o, self.bit_track(o)) for i, g in enumerate(self.tracked) for o in g.origins]

    def tracked_at(self, index: Optional[int] = None) -> List[Tuple[int, Cell]]:
        """(colour index, position) of every tracked bit at snapshot ``index``."""
        k = self.snap_index if index is None else index
        return [(i, t.position(k)) for i, _o, t in self.tracked_tracks()]

    SMALL_GROUP = 8  # groups up to this size get boxes, trails and labels; bigger ones are tinted

    def tracked_small(self, index: Optional[int] = None) -> List[Tuple[int, Cell, A.BitTrack]]:
        """Only the bits of small groups: the ones that get boxes, trails and labels."""
        k = self.snap_index if index is None else index
        out = []
        for i, g in enumerate(self.tracked):
            if len(g.origins) <= self.SMALL_GROUP:
                out.extend((i, self.bit_track(o).position(k), self.bit_track(o)) for o in g.origins)
        return out

    def tracked_big(self, index: Optional[int] = None) -> List[Tuple[int, Cell]]:
        """(colour index, position) of the bits of big groups at snapshot ``index``."""
        k = self.snap_index if index is None else index
        out = []
        for i, g in enumerate(self.tracked):
            if len(g.origins) > self.SMALL_GROUP:
                out.extend((i, self.bit_track(o).position(k)) for o in g.origins)
        return out

    # ------------------------------------------------------------ describing

    def position_text(self) -> str:
        s = self.snapshot
        pc = self.perm
        head = f"permutation {pc.index + 1}/{self.run.num_perm_calls} ({pc.phase} block {pc.block_index})"
        if s.step == "load":
            info = s.info or {}
            if info.get("phase") == "seed":
                return f"{head} · incoming block on the bus (seed), nothing loaded yet"
            if info.get("phase") == "iv":
                return f"{head} · state register before the absorb (initial value)"
            w = info.get("words", [])
            ws = f"word {w[0]}" if len(w) == 1 else f"words {w[0]}–{w[-1]}"
            return (f"{head} · loading: bus cycle {info.get('cycle')}/{info.get('n_cycles')} "
                    f"({ws} of {info.get('n_words')}, {info.get('word_bits')}-bit words)")
        if s.step == "initial":
            return f"{head} · initial state (block loaded)"
        skip = "  [disabled]" if s.skipped else ""
        return f"{head} · round {s.round + 1}/{self.params.num_rounds} · {K.STEP_SYMBOLS[s.step]} {s.step}{skip}"
