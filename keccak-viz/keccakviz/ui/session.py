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
    show_squeeze: bool = True  # append the squeeze read-out frame after calls whose output is read
    extra_inputs: Tuple[bytes, ...] = ()  # more input absorbed after the message (duplex-style)
    squeeze_sizes: Tuple[int, ...] = ()  # successive squeeze calls in bytes (empty = one call of output_bytes)
    squeeze_step: int = 32  # bytes asked for by each "+ squeeze"

    def variant_obj(self) -> S.Variant:
        base = S.VARIANTS[self.variant]
        return base.with_(rate_bytes=self.rate_bytes, domain_byte=self.domain_byte)


@dataclass
class Dye:
    """A colour applied to a set of bits at one frame; from there it spreads
    forward along the dependency pattern (and on into later permutation calls)."""

    label: str
    cells: List[Cell]  # positions at the frame where the dye was applied
    color: str  # "#rrggbb"
    start_perm: int = 0
    start_core: int = 0  # index in the bare permutation trace (0 = permutation input)
    start_load: int = -1  # loading frame it was applied in, -1 if during the permutation


@dataclass
class TrackGroup:
    """A set of bits followed together under one colour."""

    label: str
    origins: List[Cell]  # positions in snapshot 0

    @property
    def single(self) -> bool:
        return len(self.origins) == 1


def _blend_distance(d: np.ndarray) -> Optional[float]:
    """Max over cells of the total-variation distance between the cell's dye
    composition and the global composition, scaled so that 1.0 = some cell holds
    a single dye only (completely unmixed) and 0.0 = every cell holds the same
    mixture.  None if some dye has not been applied yet at this frame."""
    tot = d.sum(axis=1)
    glob = d.sum(axis=0)
    if float(glob.min()) <= 0:
        return None  # a dye has not been applied yet at this frame
    if float(tot.min()) <= 0:
        return 1.0  # some cells hold no dye yet
    comp = d / tot[:, None]
    g = glob / glob.sum()
    worst = float((1.0 - g).max())  # distance of a single-dye cell from the global mix
    return float(0.5 * np.abs(comp - g).sum(axis=1).max()) / max(worst, 1e-12)


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
    regionsChanged = QtCore.pyqtSignal(bool)  # show the rate / capacity split
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
        self.show_regions = False
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
        if set(kw) == {"squeeze_step"}:  # no effect on the run: update without recomputing
            if kw["squeeze_step"] != self.params.squeeze_step:
                self.params.squeeze_step = kw["squeeze_step"]
                self.runChanged.emit()
            return
        changed = False
        if "output_bytes" in kw and kw["output_bytes"] != self.params.output_bytes:
            self.params.squeeze_sizes = ()
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
        self.params.squeeze_sizes = ()
        if v.output_bytes is not None:
            self.params.output_bytes = v.output_bytes
        elif self.params.output_bytes == 0:
            self.params.output_bytes = 32
        self.recompute()

    def _clamp_output(self) -> None:
        """Keep the output within the specification's limit for fixed-output functions."""
        p = self.params
        m = self.max_output_bytes()
        if m is None or p.output_bytes <= m:
            return
        sizes, total = [], 0
        for n in p.squeeze_sizes:
            if total >= m:
                break
            sizes.append(min(n, m - total))
            total += sizes[-1]
        p.squeeze_sizes = tuple(sizes) if len(sizes) > 1 else ()
        p.output_bytes = m

    def recompute(self) -> None:
        self._clamp_output()
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
            extra_inputs=p.extra_inputs,
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
        key = (perm_index, p.word_bits, p.words_per_cycle, p.show_load, p.show_squeeze)
        t = self._trace_cache.get(key)
        if t is None:
            call = self.run.perm_calls[perm_index]
            block = None
            if p.show_load and call.phase == "absorb":
                block = self.run.absorb_blocks[call.block_index]
            read = self.run.read_after(perm_index) if p.show_squeeze else None
            segs = list(S.squeeze_segments(self.run, read, self.squeeze_requests())) if read is not None else None
            t = self._trace_cache[key] = S.full_trace(call, block, p.word_bits, p.words_per_cycle, read, segs)
        return t

    # ------------------------------------------------------------ squeezing more

    def squeeze_requests(self) -> Tuple[int, ...]:
        """Byte counts of the successive squeeze calls (they sum to the output length)."""
        p = self.params
        return p.squeeze_sizes if p.squeeze_sizes else (p.output_bytes,)

    def max_output_bytes(self) -> Optional[int]:
        """The specification's output limit: the digest length for fixed-output
        functions (SHA3-*, Keccak-*); None for the extendable-output SHAKE functions."""
        return S.VARIANTS[self.params.variant].output_bytes

    def can_squeeze_more(self) -> bool:
        m = self.max_output_bytes()
        return m is None or self.params.output_bytes < m

    def add_squeeze(self, n_bytes: Optional[int] = None) -> int:
        """Squeeze ``n_bytes`` more output, continuing in the rate where the last call stopped
        (a permutation runs only when the rate is used up); jump to the new read-out frame.
        Fixed-output functions stop at their digest length (the call is shortened to what is
        left, and refused once the whole digest has been read); SHAKE has no limit.
        Returns the number of bytes actually squeezed (0 = refused)."""
        n = int(n_bytes or self.params.squeeze_step)
        m = self.max_output_bytes()
        if m is not None:
            n = min(n, m - self.params.output_bytes)
        if n <= 0:
            return 0
        sizes = tuple(self.squeeze_requests()) + (n,)
        self.params.squeeze_sizes = sizes
        self.params.output_bytes = sum(sizes)
        self.recompute()
        last = len(sizes) - 1
        for pi in range(self.run.num_perm_calls - 1, -1, -1):
            tr = self.trace_for(pi)
            for snap in reversed(tr.snapshots[len(tr) - tr.n_tail:]):
                if snap.info and snap.info.get("request") == last:
                    self.set_position(perm_index=pi, snap_index=snap.index)
                    return n
        return n

    def reset_squeezes(self) -> None:
        if self.params.squeeze_sizes:
            self.params.squeeze_sizes = ()
            self.recompute()

    def output_words(self, data: bytes, start_bit: int = 0) -> List[Tuple[int, int, int, int]]:
        """(word index, value, number of bits, first bit inside the word) of ``data`` read
        from state bit ``start_bit``, grouped by the state's word boundaries (word size from
        the parameters).  ``value`` holds only the covered bits, bit 0 = the first one."""
        wb = max(1, self.params.word_bits)
        out: List[Tuple[int, int, int, int]] = []
        for j in range(len(data) * 8):
            bit = (data[j // 8] >> (j % 8)) & 1
            i = start_bit + j
            w = i // wb
            if not out or out[-1][0] != w:
                out.append((w, 0, 0, i % wb))
            ww, val, nb, first = out[-1]
            out[-1] = (ww, val | (bit << nb), nb + 1, first)
        return out

    def format_words(self, data: bytes, start_bit: int = 0, limit: Optional[int] = None) -> str:
        """Output bytes as word values (same bit order as the Words tab); a word only partly
        covered by this read shows its covered bits and their range, e.g. 26b2b8ec[32:64]."""
        words = self.output_words(data, start_bit)
        wb = self.params.word_bits
        parts = []
        for _w, val, nb, first in words[:limit]:
            digits = max(1, -(-nb // 4))
            parts.append(f"{val:0{digits}x}" + ("" if nb == wb else f"[{first}:{first + nb}]"))
        more = "" if limit is None or len(words) <= limit else f" … (+{len(words) - limit})"
        return " ".join(parts) + more

    # ------------------------------------------------------------ extra input

    def add_input(self, data: bytes) -> int:
        """Absorb ``data`` after the current last permutation; returns the new call's index."""
        self.set_params(extra_inputs=tuple(self.params.extra_inputs) + (bytes(data),))
        label = f"extra {len(self.params.extra_inputs)}"
        for b in self.run.absorb_blocks:
            if b.source == label:
                self.set_position(perm_index=b.perm.index, snap_index=0)
                return b.perm.index
        return self.perm_index

    def clear_inputs(self) -> None:
        if self.params.extra_inputs:
            self.set_params(extra_inputs=())

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
        color = color or self.DYE_COLORS[len(self.dyes) % len(self.DYE_COLORS)]
        n_load = self.n_load
        in_load = self.snap_index < n_load
        self.dyes.append(Dye(label, cells, color, self.perm_index,
                             0 if in_load else self.snap_index - n_load,
                             self.snap_index if in_load else -1))
        self._dye_cache.clear()
        self.dyesChanged.emit()
        if self.color_mode != "dye":
            self.set_color_mode("dye")
        return True

    def add_region_dye(self, region: str, color: Optional[str] = None) -> bool:
        """Dye the whole rate ("input": the bits the incoming block is XORed into) or the
        whole capacity, at the permutation input of the current call."""
        if len(self.dyes) >= self.MAX_DYES:
            return False
        r_bits = self.params.rate_bytes * 8
        idx = range(0, r_bits) if region == "input" else range(r_bits, 1600)
        cells = [K.bit_coords(i) for i in idx]
        default = "#ff3b3b" if region == "input" else "#3b8bff"
        if any(d.color == default for d in self.dyes):
            default = None
        color = color or default or self.DYE_COLORS[len(self.dyes) % len(self.DYE_COLORS)]
        name = f"input block (rate, {len(cells)} bits)" if region == "input" else f"capacity ({len(cells)} bits)"
        label = f"{name} at perm {self.perm_index + 1} input"
        self.dyes.append(Dye(label, cells, color, self.perm_index, 0, -1))
        self._dye_cache.clear()
        self.dyesChanged.emit()
        if self.color_mode != "dye":
            self.set_color_mode("dye")
        return True

    def dye_blend(self, index: Optional[int] = None) -> Optional[float]:
        """How far the dye *mixture* is from uniform at a frame (0 = every cell holds the
        dyes in the same proportions, 1 = some cell holds only one dye).  Needs two or more dyes."""
        arr = self.dye_array()
        if arr is None or arr.shape[2] < 2:
            return None
        k = self.snap_index if index is None else index
        return _blend_distance(arr[min(k, len(arr) - 1)])

    def blend_frame(self, threshold: float = 0.05) -> Optional[int]:
        """First frame of the current call at which the mixture is within ``threshold``
        of uniform everywhere (None if it never gets there in this call)."""
        arr = self.dye_array()
        if arr is None or arr.shape[2] < 2:
            return None
        for i in range(len(arr)):
            d = _blend_distance(arr[i])
            if d is not None and d < threshold:
                return i
        return None

    def dye_full_dependency(self, k: int) -> Optional[int]:
        """First frame of the current call at which *every* cell depends on *every*
        bit of dye ``k`` (exact, from the step mappings' dependency structure).
        None if that does not happen in this call or the dye was applied in an
        earlier call."""
        if not 0 <= k < len(self.dyes):
            return None
        d = self.dyes[k]
        if d.start_perm != self.perm_index:
            return None
        key = ("full", self.perm_index, k, d.start_core, d.start_load, tuple(d.cells),
               self.params.word_bits, self.params.words_per_cycle, self.params.show_load)
        if key in self._dye_cache:
            return self._dye_cache[key]
        from . import anim

        tr = self.trace
        start = (min(d.start_load, max(0, tr.n_load - 1)) if d.start_load >= 0
                 else min(tr.n_load + d.start_core, len(tr) - 1))
        src = np.array([anim.cell_index(*c) for c in d.cells])
        reach = np.zeros((len(src), 1600), dtype=bool)
        reach[np.arange(len(src)), src] = True
        found = None
        if reach.all():
            found = start
        else:
            for i in range(start + 1, len(tr)):
                snap = tr[i]
                if snap.step in K.STEP_NAMES and not snap.skipped:
                    reach = reach[:, A.source_table(snap.step)].any(axis=2)
                    if reach.all():
                        found = i
                        break
        self._dye_cache[key] = found
        return found

    def frame_label(self, i: int) -> str:
        snap = self.trace[i]
        if snap.step in K.STEP_NAMES:
            return f"round {snap.round + 1} after {K.STEP_SYMBOLS[snap.step]} {snap.step}"
        return snap.label

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

    def dye_array(self, perm_index: Optional[int] = None) -> Optional[np.ndarray]:
        """(n_snapshots, 1600, K) dye concentrations for a permutation call, or None.

        Each dye is injected at the frame where it was applied; dye present at the
        end of one call carries into the next (the state continues, and the
        absorb XOR does not mix bits)."""
        if not self.dyes:
            return None
        p = self.perm_index if perm_index is None else perm_index
        key = (p, tuple((d.start_perm, d.start_core, d.start_load, tuple(d.cells)) for d in self.dyes),
               self.params.word_bits, self.params.words_per_cycle, self.params.show_load)
        arr = self._dye_cache.get(key)
        if arr is None:
            from . import anim

            trace = self.trace_for(p)
            k_n = len(self.dyes)
            if p > 0 and any(d.start_perm < p for d in self.dyes):
                init = self.dye_array(p - 1)[-1].copy()
            else:
                init = np.zeros((1600, k_n), dtype=np.float32)
            inject = {}
            for k, d in enumerate(self.dyes):
                if d.start_perm != p:
                    continue
                if d.start_load >= 0:
                    at = min(d.start_load, max(0, trace.n_load - 1))
                else:
                    at = min(trace.n_load + d.start_core, len(trace) - 1)
                add = inject.setdefault(at, np.zeros((1600, k_n), dtype=np.float32))
                for (x, y, z) in d.cells:
                    add[anim.cell_index(x, y, z), k] = 1.0
            arr = self._dye_cache[key] = A.propagate_dye(trace, init, inject)
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
        if float(tot.max()) <= 0:
            return None  # no dye applied yet at this frame
        cols = np.array([[int(c.color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for c in self.dyes], dtype=np.float32)
        mix = (d @ cols) / np.maximum(tot, 1e-12)[:, None]
        strength = np.sqrt(tot / float(tot.max()))
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
        if float(tot.max()) <= 0:
            return None
        cols = np.array([[int(c.color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for c in self.dyes], dtype=np.float32)
        mix = (d @ cols) / np.maximum(tot, 1e-12)[:, None]
        strength = np.sqrt(tot / max(float(tot.max()), 1e-12))
        return mix.astype(np.float32), strength.astype(np.float32)

    def dye_spread(self, index: Optional[int] = None) -> Optional[float]:
        arr = self.dye_array()
        if arr is None:
            return None
        k = self.snap_index if index is None else index
        d = arr[min(k, len(arr) - 1)]
        if float(d.sum()) <= 0:
            return None
        return A.dye_spread(d)

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

    def set_show_regions(self, on: bool) -> None:
        if bool(on) != self.show_regions:
            self.show_regions = bool(on)
            self.regionsChanged.emit(self.show_regions)

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
        if s.step == "squeeze":
            info = s.info or {}
            a = info.get("start_bit", 0)
            return (f"{head} · squeeze call {info.get('request', 0) + 1}: {info.get('nbits')} output bits, "
                    f"rate bits {a}…{a + info.get('nbits', 0) - 1}")
        if s.step == "initial":
            return f"{head} · initial state (block loaded)"
        skip = "  [disabled]" if s.skipped else ""
        return f"{head} · round {s.round + 1}/{self.params.num_rounds} · {K.STEP_SYMBOLS[s.step]} {s.step}{skip}"
