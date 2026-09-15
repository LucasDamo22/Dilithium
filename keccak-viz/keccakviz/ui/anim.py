"""Per-frame geometry for the 3D cube: where every cell is and what colour
it has, as a function of the step being animated and a time t in [0, 1].

Pure NumPy; no GL here so it can be unit-tested.

Cell ordering: index i = 320*x + 64*y + z, i.e. ``bits.reshape(-1)`` of a
(5, 5, 64) array.  World position of cell (x, y, z):

    X = x - 2         (rows run along X; a row is 5 cells)
    Y = y - 2         (columns run along Y)
    Z = 31.5 - z      (lanes run along Z; z = 0 is nearest the default camera)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..core import keccak as K

XS, YS, ZS = np.meshgrid(np.arange(5), np.arange(5), np.arange(64), indexing="ij")
XS, YS, ZS = XS.reshape(-1), YS.reshape(-1), ZS.reshape(-1)
N_CELLS = 1600

BASE_POS = np.stack([XS - 2.0, YS - 2.0, 31.5 - ZS], axis=1).astype(np.float32)

SCALE_ONE = 0.78
SCALE_ZERO = 0.24

# palette (r, g, b)
COL_ONE = np.array([1.0, 0.74, 0.16])
COL_ZERO = np.array([0.26, 0.28, 0.34])
COL_CHANGED_TO_ONE = np.array([0.98, 0.32, 0.20])
COL_CHANGED_TO_ZERO = np.array([0.30, 0.62, 1.0])
COL_UNCHANGED_ONE = np.array([0.72, 0.60, 0.30])
COL_DIFF = np.array([1.0, 0.25, 0.75])
COL_DIFF_DIM_ONE = np.array([0.45, 0.40, 0.28])
COL_PULSE = np.array([1.0, 1.0, 1.0])
COL_C = np.array([0.35, 0.85, 0.95])
COL_D = np.array([0.65, 0.55, 1.0])
COL_SKIP = np.array([0.5, 0.5, 0.5])
COL_GHOST = np.array([0.75, 0.95, 1.0])
COL_EQ_ONE = np.array([1.0, 0.55, 0.15])
COL_EQ_ZERO = np.array([0.25, 0.45, 0.95])

# cell styles: (colour of 1, colour of 0, scale of 1, scale of 0)
STYLES = {
    "cubes": (COL_ONE, COL_ZERO, SCALE_ONE, SCALE_ZERO),
    "equal": (COL_EQ_ONE, COL_EQ_ZERO, 0.62, 0.62),
    "ones": (COL_ONE, COL_ZERO, SCALE_ONE, 0.0),
    "spheres": (COL_ONE, COL_ZERO, 0.9, 0.3),
    "mono": (np.array([0.97, 0.97, 0.97]), np.array([0.08, 0.08, 0.1]), 0.66, 0.66),
}

STRUCTURE_COLORS = {
    "row": (1.0, 0.35, 0.35),
    "column": (0.35, 1.0, 0.45),
    "lane": (0.4, 0.6, 1.0),
    "slice": (1.0, 0.85, 0.3),
    "plane": (1.0, 0.5, 1.0),
    "sheet": (0.4, 1.0, 1.0),
}


def cell_index(x: int, y: int, z: int) -> int:
    return 320 * x + 64 * y + z


def world_pos(x: int, y: int, z: int) -> np.ndarray:
    return np.array([x - 2.0, y - 2.0, 31.5 - z])


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def seg(t: float, a: float, b: float) -> float:
    """Map t in [a, b] to [0, 1], clamped."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return min(1.0, max(0.0, (t - a) / (b - a)))


def structure_cells(name: str, x: int, y: int, z: int) -> np.ndarray:
    """Boolean mask (1600,) of the named substructure through cell (x,y,z)."""
    if name == "row":
        return (YS == y) & (ZS == z)
    if name == "column":
        return (XS == x) & (ZS == z)
    if name == "lane":
        return (XS == x) & (YS == y)
    if name == "slice":
        return ZS == z
    if name == "plane":
        return YS == y
    if name == "sheet":
        return XS == x
    raise ValueError(name)


def structure_bounds(name: str, x: int, y: int, z: int) -> Tuple[np.ndarray, np.ndarray]:
    m = structure_cells(name, x, y, z)
    p = BASE_POS[m]
    return p.min(axis=0) - 0.5, p.max(axis=0) + 0.5


# --------------------------------------------------------------------------


@dataclass
class Frame:
    pos: np.ndarray  # (N,3) float32
    col: np.ndarray  # (N,3)
    scale: np.ndarray  # (N,)
    extra_pos: Optional[np.ndarray] = None  # extra instances (theta sheets)
    extra_col: Optional[np.ndarray] = None
    extra_scale: Optional[np.ndarray] = None
    lines: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]] = field(default_factory=list)
    # each entry: (starts (M,3), ends (M,3), rgba); overlay_lines skip the depth test
    overlay_lines: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]] = field(default_factory=list)
    # filled triangles drawn without depth test: (vertices (3M,3), rgba)
    overlay_tris: List[Tuple[np.ndarray, Tuple[float, float, float, float]]] = field(default_factory=list)
    alpha: Optional[np.ndarray] = None  # per-cell alpha (None = opaque)

    def instance_data(self) -> np.ndarray:
        n = len(self.pos)
        d = np.empty((n, 8), dtype=np.float32)
        d[:, :3] = self.pos
        d[:, 3:6] = self.col
        d[:, 6] = 1.0 if self.alpha is None else self.alpha
        d[:, 7] = self.scale
        if self.extra_pos is not None and len(self.extra_pos):
            e = np.empty((len(self.extra_pos), 8), dtype=np.float32)
            e[:, :3] = self.extra_pos
            e[:, 3:6] = self.extra_col
            e[:, 6] = 1.0
            e[:, 7] = self.extra_scale
            d = np.concatenate([d, e], axis=0)
        return d


def static_colors(cur: np.ndarray, prev: np.ndarray, mode: str,
                  diff: Optional[np.ndarray] = None, style: str = "cubes") -> Tuple[np.ndarray, np.ndarray]:
    """Colours and scales for a resting state.  ``cur``/``prev`` are flat (1600,) bit arrays."""
    cur_b = cur.astype(bool)
    c1, c0, s1, s0 = STYLES[style]
    col = np.where(cur_b[:, None], c1, c0)
    scale = np.where(cur_b, s1, s0)
    if mode == "changed":
        ch = cur_b != prev.astype(bool)
        col = np.where(ch[:, None], np.where(cur_b[:, None], COL_CHANGED_TO_ONE, COL_CHANGED_TO_ZERO),
                       np.where(cur_b[:, None], c1 * 0.72 + 0.05, c0))
        scale = np.where(ch, np.maximum(s1, 0.62), np.where(cur_b, s1 * 0.7, s0))
    elif mode == "avalanche" and diff is not None:
        d = diff.astype(bool)
        col = np.where(d[:, None], COL_DIFF, np.where(cur_b[:, None], c1 * 0.5, c0))
        scale = np.where(d, np.maximum(s1, 0.62), np.where(cur_b, s1 * 0.55, s0))
    return col.astype(np.float32), scale.astype(np.float32)


def _lerp(a, b, t):
    return a + (b - a) * t


def build_frame(step: str, prev_bits: np.ndarray, cur_bits: np.ndarray, t: float,
                mode: str, diff: Optional[np.ndarray] = None,
                skipped: bool = False, theta_c: Optional[np.ndarray] = None,
                theta_d: Optional[np.ndarray] = None, show_lines: bool = True,
                prev_colors: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                style: str = "cubes") -> Frame:
    """Geometry at time t of the transition prev -> cur performed by ``step``.

    t = 0 shows the previous state, t = 1 the current one.  ``prev_bits`` and
    ``cur_bits`` are flat (1600,) arrays.  ``prev_colors`` are the resting
    (colour, scale) arrays of the previous snapshot in the current colour
    mode, so an animation starts from what was on screen; raw colours are
    used when omitted."""
    prev_b = prev_bits.astype(bool)
    cur_b = cur_bits.astype(bool)
    if prev_colors is None:
        col_prev, sc_prev = static_colors(prev_bits, prev_bits, "raw", style=style)
    else:
        col_prev, sc_prev = prev_colors
    col_cur, sc_cur = static_colors(cur_bits, prev_bits, mode, diff, style)
    pos = BASE_POS.copy()
    fr = Frame(pos, col_cur.copy(), sc_cur.copy())

    if t >= 1.0 or step == "initial" or skipped:
        return fr
    if t <= 0.0:
        fr.col, fr.scale = col_prev, sc_prev
        return fr

    if step == "rho":
        r = K.RHO_OFFSETS[XS, YS]
        zt = (ZS + r * smoothstep(t)) % 64
        pos[:, 2] = 31.5 - zt
        # shrink near the wrap boundary so cells visibly leave one end and re-enter the other
        edge = np.minimum(zt, 64 - zt)  # distance to the seam (in cells)
        att = np.clip(edge / 0.5, 0.0, 1.0)
        # the bit travels with its cell: colour by previous value at the origin
        blend = seg(t, 0.8, 1.0)
        # at the end, the cell at origin z has moved to z+r; final colour belongs to destination
        col = _lerp(col_prev, col_cur[cell_index(XS, YS, (ZS + r) % 64)], blend)
        sc = _lerp(sc_prev, sc_cur[cell_index(XS, YS, (ZS + r) % 64)], blend) * att
        fr.col, fr.scale = col.astype(np.float32), sc.astype(np.float32)
        return fr

    if step == "pi":
        tx, ty = YS, (2 * XS + 3 * YS) % 5
        s = smoothstep(t)
        pos[:, 0] = _lerp(XS - 2.0, tx - 2.0, s)
        pos[:, 1] = _lerp(YS - 2.0, ty - 2.0, s)
        blend = seg(t, 0.8, 1.0)
        dest = cell_index(tx, ty, ZS)
        col = _lerp(col_prev, col_cur[dest], blend)
        sc = _lerp(sc_prev, sc_cur[dest], blend)
        fr.col, fr.scale = col.astype(np.float32), sc.astype(np.float32)
        _ghost_crossing_lanes(fr, s)
        if show_lines:
            fr.overlay_tris.extend(pi_arrow_tris(s))
        return fr

    flips = prev_b != cur_b

    if step == "theta" and theta_c is not None and theta_d is not None:
        cb = K.lanes_to_bits(np.asarray(theta_c)).reshape(5, 64)  # [x, z]
        db = K.lanes_to_bits(np.asarray(theta_d)).reshape(5, 64)
        gx, gz = np.meshgrid(np.arange(5), np.arange(64), indexing="ij")
        gx, gz = gx.reshape(-1), gz.reshape(-1)
        y_c, y_d = -4.2, -6.0
        t1, t2, t3 = seg(t, 0.0, 0.3), seg(t, 0.3, 0.6), seg(t, 0.6, 1.0)
        c_pos = np.stack([gx - 2.0, np.full(320, y_c), 31.5 - gz], axis=1)
        d_pos = np.stack([gx - 2.0, np.full(320, y_d), 31.5 - gz], axis=1)
        c_on = cb[gx, gz].astype(bool)
        d_on = db[gx, gz].astype(bool)
        c_col = np.where(c_on[:, None], COL_C, COL_ZERO * 0.8)
        d_col = np.where(d_on[:, None], COL_D, COL_ZERO * 0.8)
        c_sc = np.where(c_on, 0.7, 0.2) * smoothstep(t1)
        d_sc = np.where(d_on, 0.7, 0.2) * smoothstep(t2)
        fr.extra_pos = np.concatenate([c_pos, d_pos]).astype(np.float32)
        fr.extra_col = np.concatenate([c_col, d_col]).astype(np.float32)
        fr.extra_scale = np.concatenate([c_sc, d_sc]).astype(np.float32)
        # phase 1: columns collapse into C - beams from the column bottom to the C cell
        if show_lines:
            if t1 > 0 and t2 == 0:
                starts = np.stack([gx - 2.0, np.full(320, -2.5), 31.5 - gz], axis=1)
                ends = c_pos.copy()
                ends[:, 1] = -2.5 + (y_c + 0.35 + 2.5) * smoothstep(t1)
                fr.lines.append((starts, ends, (0.35, 0.85, 0.95, 0.35)))
            # phase 2: D[x] = C[x-1] ^ rot(C[x+1], 1)
            if 0 < t2 and t3 == 0:
                a = np.stack([(gx - 1) % 5 - 2.0, np.full(320, y_c - 0.35), 31.5 - gz], axis=1)
                b = np.stack([(gx + 1) % 5 - 2.0, np.full(320, y_c - 0.35), 31.5 - (gz - 1) % 64], axis=1)
                dd = d_pos + np.array([0, 0.35, 0])
                dd = a + (dd - a) * smoothstep(t2)
                fr.lines.append((a, dd, (0.65, 0.55, 1.0, 0.35)))
                dd2 = b + (d_pos + np.array([0, 0.35, 0]) - b) * smoothstep(t2)
                fr.lines.append((b, dd2, (0.65, 0.55, 1.0, 0.35)))
            # phase 3: D beams rise through the columns that flip
            if t3 > 0:
                m = d_on
                starts = d_pos[m] + np.array([0, 0.35, 0])
                ends = starts.copy()
                ends[:, 1] = -6.0 + (2.5 + 6.0) * smoothstep(t3)
                fr.lines.append((starts, ends, (0.65, 0.55, 1.0, 0.45)))
        # cells flip during phase 3
        b3 = smoothstep(t3)
        pulse = np.sin(np.pi * b3) * 0.25
        col = _lerp(col_prev, col_cur, b3)
        col = np.where(flips[:, None], _lerp(col, COL_PULSE, pulse), col)
        sc = _lerp(sc_prev, sc_cur, b3)
        sc = np.where(flips, sc + pulse, sc)
        fr.col, fr.scale = col.astype(np.float32), sc.astype(np.float32)
        return fr

    if step in ("chi", "iota", "theta"):
        # generic "cells flip" animation: pulse then recolour
        p1, p2 = seg(t, 0.0, 0.5), seg(t, 0.5, 1.0)
        pulse = np.sin(np.pi * smoothstep(p1)) * 0.3 if p2 == 0 else 0.0
        col = _lerp(col_prev, col_cur, smoothstep(p2))
        col = np.where(flips[:, None], _lerp(col, COL_PULSE, pulse), col)
        sc = _lerp(sc_prev, sc_cur, smoothstep(p2))
        sc = np.where(flips, sc + pulse, sc)
        fr.col, fr.scale = col.astype(np.float32), sc.astype(np.float32)
        if step == "chi" and show_lines and p2 < 1.0:
            # every flipping cell is driven by ~a[x+1] & a[x+2]: draw both feeds
            idx = np.nonzero(flips)[0]
            if len(idx):
                x, y, z = XS[idx], YS[idx], ZS[idx]
                tgt = BASE_POS[idx]
                s1 = BASE_POS[cell_index((x + 1) % 5, y, z)]
                s2 = BASE_POS[cell_index((x + 2) % 5, y, z)]
                alpha = 0.5 * (1 - smoothstep(p2))
                fr.lines.append((s1, tgt, (1.0, 0.4, 0.4, alpha)))
                fr.lines.append((s2, tgt, (1.0, 0.7, 0.3, alpha)))
        return fr

    return fr


def lane_crossing_strength(s: float, full: float = 0.2, none: float = 0.55) -> np.ndarray:
    """How deeply each lane (5x5 float in [0, 1], [x, y]) is passing through
    another lane's cells at progress s of the pi move.  Every cell of a lane
    moves together in the x-y plane, so two lanes 'phase' through each other
    when their interpolated (x, y) positions come close: 1 = centres within
    ``full`` of each other, 0 = further apart than ``none``."""
    gx, gy = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    tx, ty = gy, (2 * gx + 3 * gy) % 5
    px = gx + (tx - gx) * s
    py = gy + (ty - gy) * s
    d = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    np.fill_diagonal(d, np.inf)
    return np.clip((none - d.min(axis=1)) / (none - full), 0.0, 1.0).reshape(5, 5)


def crossing_lanes(s: float, threshold: float = 0.5) -> np.ndarray:
    """Boolean form of :func:`lane_crossing_strength` (strength above ``threshold``)."""
    return lane_crossing_strength(s) > threshold


def _ghost_crossing_lanes(fr: Frame, s: float) -> None:
    """Fade the cells of lanes that are crossing another lane into translucent ghosts."""
    if s <= 0.0 or s >= 1.0:
        return
    k = lane_crossing_strength(s)[XS, YS] * (fr.scale[:1600] > 0.05)
    if not (k > 0).any():
        return
    kk = k[:, None]
    fr.col[:1600] = fr.col[:1600] * (1 - 0.65 * kk) + COL_GHOST * (0.65 * kk)
    fr.scale[:1600] = fr.scale[:1600] * (1 - 0.25 * k)
    alpha = np.ones(len(fr.pos), dtype=np.float32)
    alpha[:1600] = 1.0 - 0.45 * k
    fr.alpha = alpha


def _arrow_quad(a: np.ndarray, b: np.ndarray, width: float, head: float, normal: np.ndarray) -> np.ndarray:
    """Filled arrow from a to b lying in the plane with the given normal: 3 triangles (9 verts)."""
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.zeros((0, 3))
    d = d / n
    side = np.cross(normal, d)
    side = side / max(1e-6, np.linalg.norm(side))
    tip = b
    base = b - d * min(head, n)
    w = side * (width / 2)
    hw = side * width
    tri = [
        a + w, a - w, base - w,
        a + w, base - w, base + w,
        base + hw, base - hw, tip,
    ]
    return np.array(tri)


def pi_arrow_tris(progress: float = 1.0, lanes=None, width: float = 0.2):
    """Thick filled arrows for pi on the front (z=0) and back faces, one per moving lane."""
    out = []
    normal = np.array([0.0, 0.0, 1.0])
    for x in range(5):
        for y in range(5):
            tx, ty = K.pi_target(x, y)
            if (tx, ty) == (x, y) or (lanes is not None and (x, y) not in lanes):
                continue
            a = np.array([x - 2.0, y - 2.0, 0.0])
            b = np.array([tx - 2.0, ty - 2.0, 0.0])
            hue = (x * 5 + y) / 25.0
            col = (0.5 + 0.5 * np.cos(6.283 * hue), 0.5 + 0.5 * np.cos(6.283 * (hue + 0.33)),
                   0.5 + 0.5 * np.cos(6.283 * (hue + 0.66)), 0.9)
            for zz in (32.7, -32.7):
                p0 = a + [0, 0, zz]
                p1 = a + (b - a) * max(progress, 0.05) + [0, 0, zz]
                tri = _arrow_quad(p0, p1, width, 0.45, normal)
                if len(tri):
                    out.append((tri, col))
    return out


def pi_arrows(progress: float = 1.0, lanes=None):
    """One arrow per lane on the front (z=0) and back faces: (x,y) -> (y, 2x+3y).
    ``lanes`` restricts the arrows to a set of (x, y)."""
    out = []
    zf, zb = 32.6, -32.6
    for x in range(5):
        for y in range(5):
            tx, ty = K.pi_target(x, y)
            if (tx, ty) == (x, y) or (lanes is not None and (x, y) not in lanes):
                continue
            a = np.array([x - 2.0, y - 2.0, 0.0])
            b = np.array([tx - 2.0, ty - 2.0, 0.0])
            hue = (x * 5 + y) / 25.0
            col = (0.5 + 0.5 * np.cos(6.283 * hue), 0.5 + 0.5 * np.cos(6.283 * (hue + 0.33)),
                   0.5 + 0.5 * np.cos(6.283 * (hue + 0.66)), 0.8)
            for zz in (zf, zb):
                p0 = a + [0, 0, zz]
                p1 = a + (b - a) * progress + [0, 0, zz]
                out.append((p0[None], p1[None], col))
                # arrow head at the moving end
                d = (b - a)
                n = np.linalg.norm(d)
                if n > 0:
                    d = d / n
                    side = np.array([-d[1], d[0], 0.0])
                    h1 = p1 - d * 0.35 + side * 0.2
                    h2 = p1 - d * 0.35 - side * 0.2
                    out.append((p1[None], h1[None], col))
                    out.append((p1[None], h2[None], col))
    return out


def feed_lines(step: str, x: int, y: int, z: int, cur_pos: np.ndarray):
    """Lines from the cells that feed (x, y, z) in ``step`` to that cell."""
    srcs = K.sources_of(step, x, y, z)
    tgt = cur_pos[cell_index(x, y, z)]
    starts = np.array([cur_pos[cell_index(*s)] for s in srcs])
    ends = np.repeat(tgt[None, :], len(srcs), axis=0)
    return starts, ends, srcs


def box_lines(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """12 edges of an axis-aligned box."""
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return c[[a for a, _ in e]], c[[b for _, b in e]]
