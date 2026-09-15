"""The 3D cube view: 1600 instanced cells drawn with ModernGL inside a
``QOpenGLWidget``, with QPainter overlays for labels and the HUD.

One draw call renders all cells (plus the theta C/D sheets); further draw
calls render lines (axes, highlight boxes, feed lines) and filled overlay
triangles (pi arrows).  Per frame we upload one (N, 8) float32 instance
buffer computed by :mod:`anim`; the step clock is the shared
:class:`StepAnimator`.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..core import keccak as K
from . import anim
from .animator import StepAnimator
from .camera import OrbitCamera
from .session import Session

try:
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None


def install_surface_format() -> None:
    """Must be called before the QApplication is created."""
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


VERT_CELL = """
#version 330
uniform mat4 mvp;
uniform vec3 light_dir;
uniform vec3 spacing;
in vec3 in_pos;
in vec3 in_norm;
in vec3 inst_pos;
in vec4 inst_col;
in float inst_scale;
out vec4 v_col;
void main() {
    vec3 p = inst_pos * spacing + in_pos * inst_scale;
    gl_Position = mvp * vec4(p, 1.0);
    float l = 0.62 + 0.38 * max(dot(in_norm, normalize(light_dir)), 0.0);
    v_col = vec4(inst_col.rgb * l, inst_col.a);
}
"""
FRAG_PLAIN = """
#version 330
in vec4 v_col;
out vec4 f_col;
void main() { f_col = v_col; }
"""
VERT_LINE = """
#version 330
uniform mat4 mvp;
uniform vec3 spacing;
in vec3 in_pos;
in vec4 in_col;
out vec4 v_col;
void main() { gl_Position = mvp * vec4(in_pos * spacing, 1.0); v_col = in_col; }
"""


def _rgb(hexcol: str) -> Tuple[float, float, float]:
    c = QtGui.QColor(hexcol)
    return c.redF(), c.greenF(), c.blueF()


def unit_cube_mesh() -> np.ndarray:
    """36 vertices (pos3, normal3) of a cube spanning [-0.5, 0.5]^3."""
    verts = []
    for axis in range(3):
        for s in (-1, 1):
            n = np.zeros(3)
            n[axis] = s
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1
            v = np.zeros(3)
            v[(axis + 2) % 3] = 1
            c = n * 0.5
            a, b, cc, d = c - u * 0.5 - v * 0.5, c + u * 0.5 - v * 0.5, c + u * 0.5 + v * 0.5, c - u * 0.5 + v * 0.5
            tri = [a, b, cc, a, cc, d] if s > 0 else [a, cc, b, a, d, cc]
            for p in tri:
                verts.append(np.concatenate([p, n]))
    return np.array(verts, dtype=np.float32)


def unit_sphere_mesh(rings: int = 8, segs: int = 12) -> np.ndarray:
    """UV sphere of diameter 1 as a triangle list (pos3, normal3)."""
    verts = []

    def pt(t, p):
        return np.array([math.sin(t) * math.cos(p), math.cos(t), math.sin(t) * math.sin(p)])

    for i in range(rings):
        t0, t1 = math.pi * i / rings, math.pi * (i + 1) / rings
        for j in range(segs):
            p0, p1 = 2 * math.pi * j / segs, 2 * math.pi * (j + 1) / segs
            a, b, c, d = pt(t0, p0), pt(t1, p0), pt(t1, p1), pt(t0, p1)
            for tri in ((a, b, c), (a, c, d)):
                for n in tri:
                    verts.append(np.concatenate([n * 0.5, n]))
    return np.array(verts, dtype=np.float32)


class CubeView(QtWidgets.QOpenGLWidget):
    """Interactive 3D view of the current state."""

    cellHovered = QtCore.pyqtSignal(object)
    fpsMeasured = QtCore.pyqtSignal(float)

    LINE_MODES = ("none", "focused", "all")
    MAX_INSTANCES = 1600 + 640
    MAX_LINE_VERTS = 40000
    AXES_ORIGIN = np.array([-3.4, -3.4, 33.4])

    def __init__(self, session: Session, animator: StepAnimator, parent=None):
        super().__init__(parent)
        self.session = session
        self.animator = animator
        self.camera = OrbitCamera(distance=100.0, target=np.array([0.0, 0.0, 5.0]))
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)

        self.ctx = None
        self._frame: Optional[anim.Frame] = None
        self._pair: Tuple[Optional[K.Snapshot], Optional[K.Snapshot], float] = (None, None, 1.0)
        self.line_mode = "focused"  # none / focused (selected + tracked bits) / all
        self.show_labels = True
        self.show_axes = True
        self.show_formula = True
        self.show_sheets = True
        self.show_trails = True
        self.show_bus = True
        self.show_word_bands = False
        self.show_tracked = True
        self.show_dye_legend = True
        self.spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self._last_pos: Tuple[int, int] = (0, 0)
        self._press_pos: Tuple[int, int] = (0, 0)
        self._hover: Optional[Tuple[int, int, int]] = None
        self._fps_t0 = time.perf_counter()
        self._fps_n = 0
        self.fps = 0.0

        animator.changed.connect(self.update)
        session.runChanged.connect(self.update)
        session.selectionChanged.connect(lambda _c: self.update())
        session.structureChanged.connect(lambda _s: self.update())
        session.colorModeChanged.connect(lambda _m: self.update())
        session.styleChanged.connect(lambda _m: self.update())
        session.visibilityChanged.connect(lambda _m: self.update())
        session.dyesChanged.connect(self.update)
        session.pulledChanged.connect(self.update)
        session.trackedChanged.connect(self.update)

    # ------------------------------------------------------------ GL setup

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(vertex_shader=VERT_CELL, fragment_shader=FRAG_PLAIN)
        self.line_prog = self.ctx.program(vertex_shader=VERT_LINE, fragment_shader=FRAG_PLAIN)
        self.mesh_vbos = {"cube": self.ctx.buffer(unit_cube_mesh().tobytes()),
                          "sphere": self.ctx.buffer(unit_sphere_mesh().tobytes())}
        self.inst = self.ctx.buffer(reserve=self.MAX_INSTANCES * 8 * 4, dynamic=True)
        self.vaos = {
            name: self.ctx.vertex_array(
                self.prog,
                [(vbo, "3f 3f", "in_pos", "in_norm"),
                 (self.inst, "3f 4f 1f/i", "inst_pos", "inst_col", "inst_scale")])
            for name, vbo in self.mesh_vbos.items()
        }
        self.line_vbo = self.ctx.buffer(reserve=self.MAX_LINE_VERTS * 7 * 4, dynamic=True)
        self.line_vao = self.ctx.vertex_array(self.line_prog, [(self.line_vbo, "3f 4f", "in_pos", "in_col")])
        self.prog["light_dir"].value = (0.4, 0.9, 0.7)

    # ------------------------------------------------------------ frame

    @property
    def animating(self) -> bool:
        return self.animator.active

    def freeze_animation(self, t: float) -> None:
        self.animator.freeze(t)

    def current_frame(self) -> anim.Frame:
        s = self.session
        tr = s.trace
        mode = s.color_mode
        style = s.cell_style
        prev, snap, t = self.animator.pair()
        self._pair = (prev, snap, t)
        prev_bits = K.lanes_to_bits(prev.state).reshape(-1)
        cur_bits = K.lanes_to_bits(snap.state).reshape(-1)
        diff = prev_diff = None
        if mode == "avalanche":
            av = s.avalanche()
            diff = av.diff_bits(tr.core_index(snap.index)).reshape(-1)
            prev_diff = av.diff_bits(tr.core_index(prev.index)).reshape(-1)
        prev_prev = tr[max(0, prev.index - 1)]
        prev_prev_bits = K.lanes_to_bits(prev_prev.state).reshape(-1)
        dye_prev = dye_cur = None
        if mode == "dye":
            dye_prev = s.dye_render(prev.index)
            dye_cur = s.dye_render(snap.index)
        prev_colors = anim.static_colors(prev_bits, prev_prev_bits, mode, prev_diff, style, dye_prev)
        cur_colors = anim.static_colors(cur_bits, prev_bits, mode, diff, style, dye_cur)
        load_cells = None
        if snap.step == "load" and snap.info and snap.info.get("cells"):
            load_cells = np.array([anim.cell_index(*K.bit_coords(i)) for i in snap.info["cells"]])
        return anim.build_frame(snap.step, prev_bits, cur_bits, t, mode, diff, snap.skipped,
                                snap.theta_c, snap.theta_d, self.line_mode == "all", prev_colors, style,
                                cur_colors, load_cells, s.pull_offsets(), self.show_sheets, self.show_bus)

    def _in_transition(self) -> bool:
        prev, snap, t = self._pair
        return snap is not None and t < 1.0 and snap.step != "initial"

    def _layout_index(self) -> int:
        """Snapshot whose positions the frame's cells are indexed by."""
        prev, snap, t = self._pair
        return prev.index if self._in_transition() else self.session.snap_index

    def _tracked_prev_layout(self):
        return self.session.tracked_at(self._layout_index())

    def tracked_world_positions(self, frame: anim.Frame):
        """(colour index, position (3,), cell) of every small-group tracked bit right now (unspaced)."""
        return [(ci, frame.pos[anim.cell_index(*cell)], cell)
                for ci, cell, _t in self.session.tracked_small(self._layout_index())]

    def _next_step(self) -> Optional[str]:
        s = self.session
        if s.snap_index + 1 < s.num_snapshots:
            return s.trace[s.snap_index + 1].step
        return None

    # ------------------------------------------------------------ drawing

    def _mvp(self) -> np.ndarray:
        aspect = self.width() / max(1, self.height())
        return self.camera.mvp(aspect)

    def paintGL(self) -> None:
        painter = QtGui.QPainter(self)
        painter.beginNativePainting()
        self._draw_gl()
        painter.endNativePainting()
        if self.show_labels:
            self._draw_overlay(painter)
        painter.end()
        self._fps_n += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t0)
            self._fps_n = 0
            self._fps_t0 = now
            self.fpsMeasured.emit(self.fps)

    def _draw_gl(self) -> None:
        ctx = self.ctx
        fbo = ctx.detect_framebuffer(self.defaultFramebufferObject())
        fbo.use()
        ratio = self.devicePixelRatioF()
        ctx.viewport = (0, 0, int(self.width() * ratio), int(self.height() * ratio))
        ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.clear(0.09, 0.10, 0.13, 1.0, depth=1.0)

        frame = self.current_frame()
        self._frame = frame
        mvp = self._mvp().astype(np.float32)
        mvp_bytes = mvp.T.tobytes()
        spacing = tuple(float(v) for v in self.spacing)

        self._apply_highlights(frame)
        data = frame.instance_data()
        self.inst.write(data.tobytes())
        self.prog["mvp"].write(mvp_bytes)
        self.prog["spacing"].value = spacing
        mesh = "sphere" if self.session.cell_style == "spheres" else "cube"
        self.vaos[mesh].render(instances=len(data))

        ctx.disable(moderngl.CULL_FACE)
        self.line_prog["mvp"].write(mvp_bytes)
        self.line_prog["spacing"].value = spacing
        verts = self._line_verts(frame, overlay=False)
        if len(verts):
            self.line_vbo.write(verts.tobytes())
            self.line_vao.render(moderngl.LINES, vertices=len(verts))
        ctx.disable(moderngl.DEPTH_TEST)
        verts = self._line_verts(frame, overlay=True)
        if len(verts):
            self.line_vbo.write(verts.tobytes())
            self.line_vao.render(moderngl.LINES, vertices=len(verts))
        # hand a clean state back to Qt's own paint engine (text is textured quads)
        ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)

    def _apply_highlights(self, frame: anim.Frame) -> None:
        """Tint the selected cell / substructure / hover cell in place."""
        s = self.session
        sel = s.selected
        anchor = sel or self._hover or (0, 0, 0)
        if s.structure:
            m = anim.structure_cells(s.structure, *anchor)
            c = np.array(anim.STRUCTURE_COLORS[s.structure])
            frame.col[m] = frame.col[m] * 0.35 + c * 0.65
            frame.scale[m] = np.maximum(frame.scale[m], 0.55)
            # push everything else back so the structure reads through other layers
            frame.col[~m] *= 0.45
            frame.scale[~m] *= 0.55
        if sel is not None:
            i = anim.cell_index(*sel)
            frame.col[i] = (1.0, 1.0, 1.0)
            frame.scale[i] = max(frame.scale[i], 0.9)
            nxt = self._next_step()
            if nxt:
                for src in K.sources_of(nxt, *sel):
                    j = anim.cell_index(*src)
                    if j != i:
                        frame.col[j] = frame.col[j] * 0.3 + np.array([0.2, 1.0, 0.9]) * 0.7
                        frame.scale[j] = max(frame.scale[j], 0.6)
        if self._hover is not None and self._hover != sel:
            i = anim.cell_index(*self._hover)
            frame.col[i] = frame.col[i] * 0.5 + 0.5
        if self.show_word_bands:
            wb = max(1, s.params.word_bits)
            word = (K.bit_index(anim.XS, anim.YS, anim.ZS) // wb) % 2
            frame.col[:1600][word == 1] *= 0.62
        k = self._layout_index()
        for _ci, cell, _t in (s.tracked_small(k) if self.show_tracked else []):
            i = anim.cell_index(*cell)
            frame.scale[i] = max(frame.scale[i], 0.7)
        big = s.tracked_big(k) if self.show_tracked else []
        if big:
            idx = np.array([anim.cell_index(*c) for _ci, c in big])
            cols = np.array([_rgb(s.TRACK_COLORS[ci]) for ci, _c in big])
            frame.col[idx] = frame.col[idx] * 0.35 + cols * 0.65
            frame.scale[idx] = np.maximum(frame.scale[idx], 0.45)
        if s.visibility != "both":
            # hide 0s or 1s; a cell's value is the one it carries in the frame's layout
            layout = s.trace[self._layout_index()]
            bits = K.lanes_to_bits(layout.state).reshape(-1)
            hide = bits == (1 if s.visibility == "zeros" else 0)
            frame.scale[:1600][hide] = 0.0

    def _line_verts(self, frame: anim.Frame, overlay: bool) -> np.ndarray:
        """Line segments; ``overlay`` ones are drawn without depth testing."""
        segs: List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]] = []
        s = self.session
        anchor = s.selected or self._hover or (0, 0, 0)
        if not overlay:
            lo = np.array([-2.5, -2.5, -32.0])
            hi = np.array([2.5, 2.5, 32.0])
            if self.show_axes:
                a, b = anim.box_lines(lo, hi)
                segs.append((a, b, (0.5, 0.5, 0.6, 0.35)))
                o = self.AXES_ORIGIN
                segs.append((o[None], (o + [5, 0, 0])[None], (1.0, 0.35, 0.35, 0.9)))
                segs.append((o[None], (o + [0, 5, 0])[None], (0.35, 1.0, 0.45, 0.9)))
                segs.append((o[None], (o + [0, 0, -10])[None], (0.4, 0.6, 1.0, 0.9)))
            segs.extend(frame.lines)
            shown = self._pair[1]
            if self.show_bus and shown is not None and shown.step == "load":
                bx = anim.BUS_OFFSET[0]
                segs.append((np.array([[bx - 2.0, -2.5, 32.0]]), np.array([[bx - 2.0, -2.5, -32.0]]),
                             tuple(anim.COL_BUS) + (0.9,)))
                segs.append((np.array([[bx + 2.0, -2.5, 32.0]]), np.array([[bx + 2.0, -2.5, -32.0]]),
                             tuple(anim.COL_BUS) + (0.9,)))
        else:
            if s.structure:
                blo, bhi = anim.structure_bounds(s.structure, *anchor)
                a, b = anim.box_lines(blo - 0.05, bhi + 0.05)
                segs.append((a, b, anim.STRUCTURE_COLORS[s.structure] + (0.95,)))
            transition = self._in_transition()
            if s.selected is not None and self.line_mode != "none" and not transition:
                nxt = self._next_step()
                if nxt:
                    a, b, _ = anim.feed_lines(nxt, *s.selected, frame.pos)
                    segs.append((a, b, (0.2, 1.0, 0.9, 0.9)))
                i = anim.cell_index(*s.selected)
                p = frame.pos[i]
                a, b = anim.box_lines(p - 0.55, p + 0.55)
                segs.append((a, b, (1.0, 1.0, 1.0, 1.0)))
            if self.show_tracked:
                segs.extend(self._tracked_lines(frame))
            off = s.pull_offsets()
            if off is not None:
                for k_, (name, cell) in enumerate(s.pulled):
                    blo, bhi = anim.structure_bounds(name, *cell)
                    lift = np.array([0.0, 8.0 + 7.0 * k_, 0.0])
                    a, b = anim.box_lines(blo - 0.15 + lift, bhi + 0.15 + lift)
                    segs.append((a, b, anim.STRUCTURE_COLORS[name] + (0.8,)))
                    # tether from the cube to the lifted region
                    c0 = (blo + bhi) / 2
                    segs.append((c0[None], (c0 + lift)[None], anim.STRUCTURE_COLORS[name] + (0.35,)))
            if self.line_mode == "all":
                segs.extend(frame.overlay_lines)
            elif self.line_mode == "focused" and transition and self._pair[1].step == "pi":
                lanes = {(c[0], c[1]) for _i, c in self._tracked_prev_layout()}
                if s.selected is not None:
                    lanes.add((s.selected[0], s.selected[1]))
                # nothing in focus: show every lane's arrow, that is the whole point of pi
                segs.extend(anim.pi_arrows(anim.smoothstep(self._pair[2]), lanes or None))
        if not segs:
            return np.zeros((0, 7), dtype=np.float32)
        parts = []
        for a, b, rgba in segs:
            n = len(a)
            v = np.empty((2 * n, 7), dtype=np.float32)
            v[0::2, :3] = a
            v[1::2, :3] = b
            v[:, 3:] = rgba
            parts.append(v)
        out = np.concatenate(parts)
        return out[: self.MAX_LINE_VERTS]

    def _tracked_lines(self, frame: anim.Frame):
        """Marker boxes, trails and (in focused mode) feed lines for tracked bits."""
        s = self.session
        segs = []
        transition = self._in_transition()
        upto = self._layout_index()
        for (ci, pos, cell), (_ci2, _cell2, track) in zip(self.tracked_world_positions(frame), s.tracked_small(upto)):
            rgb = _rgb(s.TRACK_COLORS[ci])
            a, b = anim.box_lines(pos - 0.62, pos + 0.62)
            segs.append((a, b, rgb + (1.0,)))
            pts = [anim.BASE_POS[anim.cell_index(*track.position(j))] for j in range(0, upto + 1)]
            pts.append(pos)
            pts = np.array(pts)
            if len(pts) > 1 and self.show_trails:
                d = np.abs(np.diff(pts, axis=0)).sum(axis=1)
                keep = d > 1e-6
                if keep.any():
                    segs.append((pts[:-1][keep], pts[1:][keep], rgb + (0.55,)))
            step = self._pair[1].step if transition else None
            if self.line_mode != "none" and step in ("theta", "chi"):
                srcs = K.sources_of(step, *cell)
                a = np.array([frame.pos[anim.cell_index(*c)] for c in srcs if c != cell])
                if len(a):
                    b = np.repeat(pos[None, :], len(a), axis=0)
                    segs.append((a, b, rgb + (0.8,)))
        return segs

    # ------------------------------------------------------------ overlay

    def _project(self, mvp, pts, w, h):
        return self.camera.project(mvp, np.asarray(pts, dtype=np.float64) * self.spacing, w, h)

    def _draw_overlay(self, p: QtGui.QPainter) -> None:
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        mvp = self._mvp()
        font = p.font()
        font.setPointSize(10)
        p.setFont(font)
        if self.show_axes:
            o = self.AXES_ORIGIN
            pts = np.array([o + [5.6, 0, 0], o + [0, 5.6, 0], o + [0, 0, -11], o])
            sp = self._project(mvp, pts, w, h)
            labels = [("x  (row →)", (255, 90, 90)), ("y  (column ↑)", (90, 255, 115)),
                      ("z  (lane, 64 bits)", (100, 150, 255)), ("(0,0,0)", (200, 200, 200))]
            for (sx, sy, wv), (txt, col) in zip(sp, labels):
                if wv > 0:
                    p.setPen(QtGui.QColor(*col))
                    p.drawText(QtCore.QPointF(sx + 4, sy - 4), txt)
        s = self.session
        snap = s.snapshot
        shown_snap = self._pair[1] or snap
        if self.show_bus and shown_snap.step == "load":
            bx = anim.BUS_OFFSET[0]
            sp = self._project(mvp, np.array([[bx, -2.5, 33.5]]), w, h)[0]
            if sp[2] > 0:
                p.setPen(QtGui.QColor(80, 240, 255))
                info = shown_snap.info or {}
                p.drawText(QtCore.QPointF(sp[0] - 10, sp[1] + 16),
                           f"bus ({info.get('word_bits', 64) * s.params.words_per_cycle} bits/cycle)")
        p.setPen(QtGui.QColor(235, 235, 240))
        f2 = QtGui.QFont(font)
        f2.setPointSize(12)
        f2.setBold(True)
        p.setFont(f2)
        p.drawText(12, 22, s.position_text())
        p.setFont(font)
        y = 42
        if snap.step in K.STEP_NAMES:
            p.setPen(QtGui.QColor(190, 190, 200))
            p.drawText(12, y, K.step_description(snap.step))
            y += 18
        if self.show_formula:
            # billboard above the cube: operation and formula of the step being shown
            shown = self._pair[1] or snap
            top_pt = np.array([[0.0, 3.6 + (8.0 + 7.0 * (len(s.pulled) - 1) + 3.0 if s.pulled else 0.0), 0.0]])
            sx, sy, wv = self._project(mvp, top_pt, w, h)[0]
            if wv > 0:
                f3 = QtGui.QFont(font)
                f3.setPointSize(15)
                f3.setBold(True)
                p.setFont(f3)
                if shown.step in K.STEP_NAMES:
                    title = f"{K.STEP_SYMBOLS[shown.step]}  {shown.step}"
                    formula = K.step_description(shown.step)
                elif shown.step == "load":
                    info = shown.info or {}
                    title = "⇥  load"
                    formula = {"seed": "block on the bus (seed)", "iv": "state register (initial value)"}.get(
                        info.get("phase"), f"state[word] ^= bus_word   cycle {info.get('cycle')}/{info.get('n_cycles')}")
                else:
                    title, formula = "initial state", "block loaded, permutation starts"
                tw = p.fontMetrics().horizontalAdvance(title)
                p.setPen(QtGui.QColor(255, 235, 150))
                p.drawText(QtCore.QPointF(sx - tw / 2, sy - 24), title)
                f4 = QtGui.QFont(font)
                f4.setPointSize(11)
                p.setFont(f4)
                fw = p.fontMetrics().horizontalAdvance(formula)
                p.setPen(QtGui.QColor(220, 220, 230))
                p.drawText(QtCore.QPointF(sx - fw / 2, sy - 6), formula)
                p.setFont(font)
            if snap.step == "iota" and snap.round_constant is not None:
                p.drawText(12, y, f"RC[{snap.round_index_abs}] = 0x{snap.round_constant:016x}")
                y += 18
        if snap.skipped:
            p.setPen(QtGui.QColor(255, 120, 120))
            p.drawText(12, y, "step disabled in parameters - state unchanged")
            y += 18
        y += 6
        c1, c0, _s1, _s0 = anim.STYLES[s.cell_style]
        legend = {
            "raw": [("1", c1), ("0", c0)],
            "changed": [("flipped to 1", anim.COL_CHANGED_TO_ONE), ("flipped to 0", anim.COL_CHANGED_TO_ZERO),
                        ("unchanged 1", c1 * 0.72 + 0.05), ("unchanged 0", c0)],
            "avalanche": [("differs from flipped run", anim.COL_DIFF), ("same, value 1", c1 * 0.5),
                          ("same, value 0", c0)],
            "dye": [],
        }[s.color_mode]
        if snap.step == "theta" and self.animating:
            legend = legend + [("C[x] parity sheet", anim.COL_C), ("D[x] correction sheet", anim.COL_D)]
        if s.color_mode == "dye":
            legend = [(f"dye {i + 1}: {d.label}", np.array(_rgb(d.color))) for i, d in enumerate(s.dyes)]
            legend += [("no dye yet (value colour, dimmed)", c1 * 0.45)]
            spread = s.dye_spread()
            if spread is not None:
                legend += [(f"unevenness max/mean = {spread:.2f} (1.00 = homogeneous)", np.array([0.5, 0.5, 0.5]))]
        if self._frame is not None and self._frame.alpha is not None:
            legend = legend + [("phasing through another cell", anim.COL_GHOST)]
        if snap.step == "load" and self.show_bus:
            legend = legend + [("word arriving on the bus", anim.COL_BUS)]
        for name, col in legend:
            p.fillRect(12, y - 10, 12, 12, QtGui.QColor(*(int(min(1.0, c) * 255) for c in col)))
            p.setPen(QtGui.QColor(200, 200, 210))
            p.drawText(30, y, name)
            y += 17
        cell = s.selected or self._hover
        if cell is not None:
            x, yy, z = cell
            bits = K.lanes_to_bits(snap.state)
            val = int(bits[x, yy, z])
            lines = [
                f"cell (x={x}, y={yy}, z={z})   lane {K.lane_index(x, yy)} = A[{x},{yy}]   bit index {K.bit_index(x, yy, z)}",
                f"value {val}   lane hex 0x{int(snap.state[x, yy]):016x}   rho offset {int(K.RHO_OFFSETS[x, yy])}",
            ]
            nxt = self._next_step()
            if nxt:
                srcs = K.sources_of(nxt, x, yy, z)
                lines.append(f"next step {K.STEP_SYMBOLS[nxt]} {nxt}: fed by {len(srcs)} cell(s) "
                             + ", ".join(f"({a},{b},{c})" for a, b, c in srcs[:11]))
            fy = h - 12 - 17 * (len(lines) - 1)
            p.fillRect(6, fy - 16, w - 12, 17 * len(lines) + 6, QtGui.QColor(0, 0, 0, 140))
            p.setPen(QtGui.QColor(240, 240, 245))
            for ln in lines:
                p.drawText(12, fy, ln)
                fy += 17
        if self._frame is not None and s.tracked:
            k = self._layout_index()
            small = s.tracked_small(k)
            if small:
                pts = np.array([self._frame.pos[anim.cell_index(*cell)] for _ci, cell, _t in small])
                sp = self._project(mvp, pts, w, h)
                for (ci, _cell, track), (sx, sy, wv) in zip(small, sp):
                    if wv <= 0:
                        continue
                    p.setPen(QtGui.QColor(s.TRACK_COLORS[ci]))
                    p.drawText(QtCore.QPointF(sx + 8, sy - 6), f"#{ci + 1} = {track.value(k)}")
        p.setPen(QtGui.QColor(120, 120, 130))
        p.drawText(w - 150, h - 8, f"{self.fps:4.0f} fps  |  1600 cells, 1 draw call")

    # ------------------------------------------------------------ picking

    def pick(self, px: float, py: float) -> Optional[Tuple[int, int, int]]:
        if self._frame is None:
            return None
        mvp = self._mvp()
        o, d = self.camera.ray(mvp, px, py, self.width(), self.height())
        pos = self._frame.pos[:1600].astype(np.float64) * self.spacing
        half = (np.maximum(self._frame.scale[:1600], 0.3) * 0.5)[:, None]
        lo, hi = pos - half, pos + half
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
            t0 = (lo - o) * inv
            t1 = (hi - o) * inv
        tmin = np.minimum(t0, t1).max(axis=1)
        tmax = np.maximum(t0, t1).min(axis=1)
        hit = (tmax >= tmin) & (tmax > 0)
        if not hit.any():
            return None
        tmin = np.where(hit, tmin, np.inf)
        i = int(np.argmin(tmin))
        return int(anim.XS[i]), int(anim.YS[i]), int(anim.ZS[i])

    # ------------------------------------------------------------ mouse / keys

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._last_pos = (ev.x(), ev.y())
        self._press_pos = (ev.x(), ev.y())
        self.setFocus()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        dx = ev.x() - self._last_pos[0]
        dy = ev.y() - self._last_pos[1]
        self._last_pos = (ev.x(), ev.y())
        if ev.buttons() & QtCore.Qt.LeftButton and not (ev.modifiers() & QtCore.Qt.ShiftModifier):
            self.camera.orbit(dx, dy)
            self.update()
        elif ev.buttons() & (QtCore.Qt.RightButton | QtCore.Qt.MiddleButton) or (
                ev.buttons() & QtCore.Qt.LeftButton):
            self.camera.pan(dx, dy, self.height())
            self.update()
        else:
            cell = self.pick(ev.x(), ev.y())
            if cell != self._hover:
                self._hover = cell
                self.cellHovered.emit(cell)
                self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        moved = abs(ev.x() - self._press_pos[0]) + abs(ev.y() - self._press_pos[1])
        if ev.button() == QtCore.Qt.LeftButton and moved < 4:
            self.session.select(self.pick(ev.x(), ev.y()))

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.session.select(None)

    def leaveEvent(self, ev) -> None:
        self._hover = None
        self.update()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        self.camera.zoom(ev.angleDelta().y() / 120.0)
        self.update()

    # ------------------------------------------------------------ settings

    def set_preset(self, name: str) -> None:
        self.camera.preset(name)
        self.camera.target = np.array([0.0, 0.0, 5.0]) if name == "perspective" else np.zeros(3)
        self.camera.distance = {"lane-on": 95.0, "top": 95.0, "slice-on": 40.0, "isometric": 80.0}.get(name, 100.0)
        self.camera.distance *= float(max(self.spacing))
        self.update()

    def set_spacing(self, axis: int, value: float) -> None:
        old = float(max(self.spacing))
        self.spacing[axis] = value
        new = float(max(self.spacing))
        if old > 0 and new != old:
            self.camera.distance *= new / old  # keep the whole cube in view
        self.update()

    def set_animation_ms(self, ms: int) -> None:
        self.animator.set_animation_ms(ms)

    DISPLAY_TOGGLES = (
        ("show_axes", "axes and bounding box"),
        ("show_labels", "HUD, legend and cell info"),
        ("show_formula", "operation / formula label above the cube"),
        ("show_sheets", "θ parity (C) and correction (D) sheets"),
        ("show_bus", "words flying in from the bus while loading"),
        ("show_tracked", "tracked bits (boxes, tints, labels)"),
        ("show_trails", "trails of tracked bits"),
        ("show_word_bands", "shade alternate words (word size from parameters)"),
    )

    def set_toggle(self, name: str, on: bool) -> None:
        setattr(self, name, bool(on))
        self.update()

    def set_line_mode(self, mode: str) -> None:
        self.line_mode = mode
        self.update()

    def grab_png(self, path: str) -> None:
        self.grabFramebuffer().save(path)
