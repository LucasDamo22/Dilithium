"""The 3D cube view: 1600 instanced cubes drawn with ModernGL inside a
``QOpenGLWidget``, with QPainter overlays for labels and the HUD.

One draw call renders all cells (plus the theta C/D sheets); a second draw
call renders every line (axes, highlight boxes, feed lines).  Per frame we
upload one (N, 8) float32 instance buffer computed by :mod:`anim`.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..core import keccak as K
from . import anim
from .camera import OrbitCamera, PRESETS
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


VERT_CUBE = """
#version 330
uniform mat4 mvp;
uniform vec3 light_dir;
in vec3 in_pos;
in vec3 in_norm;
in vec3 inst_pos;
in vec4 inst_col;
in float inst_scale;
out vec4 v_col;
void main() {
    vec3 p = inst_pos + in_pos * inst_scale;
    gl_Position = mvp * vec4(p, 1.0);
    float l = 0.62 + 0.38 * max(dot(in_norm, normalize(light_dir)), 0.0);
    v_col = vec4(inst_col.rgb * l, inst_col.a);
}
"""
FRAG_CUBE = """
#version 330
in vec4 v_col;
out vec4 f_col;
void main() { f_col = v_col; }
"""
VERT_LINE = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec4 in_col;
out vec4 v_col;
void main() { gl_Position = mvp * vec4(in_pos, 1.0); v_col = in_col; }
"""
FRAG_LINE = """
#version 330
in vec4 v_col;
out vec4 f_col;
void main() { f_col = v_col; }
"""


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


class CubeView(QtWidgets.QOpenGLWidget):
    """Interactive 3D view of the current state."""

    cellHovered = QtCore.pyqtSignal(object)
    fpsMeasured = QtCore.pyqtSignal(float)

    MAX_INSTANCES = 1600 + 640
    MAX_LINE_VERTS = 20000

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.camera = OrbitCamera(distance=100.0, target=np.array([0.0, 0.0, 5.0]))
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)

        self.ctx = None
        self._frame: Optional[anim.Frame] = None
        self._anim_from = 0  # snapshot index being left
        self._anim_to = 0
        self._anim_t = 1.0
        self._anim_start = 0.0
        self._anim_dir = 1
        self._frozen = False
        self.animation_ms = 900
        self.show_lines = True
        self.show_labels = True
        self.show_axes = True
        self._last_pos: Tuple[int, int] = (0, 0)
        self._press_pos: Tuple[int, int] = (0, 0)
        self._dragging = False
        self._hover: Optional[Tuple[int, int, int]] = None
        self._last_perm = session.perm_index
        self._last_snap = session.snap_index
        self._fps_t0 = time.perf_counter()
        self._fps_n = 0
        self.fps = 0.0

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

        session.positionChanged.connect(self._on_position)
        session.runChanged.connect(self._on_run)
        session.selectionChanged.connect(lambda _c: self.update())
        session.structureChanged.connect(lambda _s: self.update())
        session.colorModeChanged.connect(lambda _m: self.update())

    # ------------------------------------------------------------ GL setup

    def initializeGL(self) -> None:
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(vertex_shader=VERT_CUBE, fragment_shader=FRAG_CUBE)
        self.line_prog = self.ctx.program(vertex_shader=VERT_LINE, fragment_shader=FRAG_LINE)
        self.vbo = self.ctx.buffer(unit_cube_mesh().tobytes())
        self.inst = self.ctx.buffer(reserve=self.MAX_INSTANCES * 8 * 4, dynamic=True)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f", "in_pos", "in_norm"),
             (self.inst, "3f 4f 1f/i", "inst_pos", "inst_col", "inst_scale")],
        )
        self.line_vbo = self.ctx.buffer(reserve=self.MAX_LINE_VERTS * 7 * 4, dynamic=True)
        self.line_vao = self.ctx.vertex_array(self.line_prog, [(self.line_vbo, "3f 4f", "in_pos", "in_col")])
        self.prog["light_dir"].value = (0.4, 0.9, 0.7)

    def resizeGL(self, w: int, h: int) -> None:
        pass

    # ------------------------------------------------------------ animation

    def _on_run(self) -> None:
        self._last_perm = self.session.perm_index
        self._last_snap = self.session.snap_index
        self._anim_t = 1.0
        self._timer.stop()
        self.update()

    def _on_position(self, perm: int, snap: int) -> None:
        self._frozen = False
        if perm == self._last_perm and snap == self._last_snap + 1:
            self._start_anim(snap, +1)
        elif perm == self._last_perm and snap == self._last_snap - 1:
            self._start_anim(self._last_snap, -1)
        else:
            self._anim_t = 1.0
            self._timer.stop()
        self._last_perm, self._last_snap = perm, snap
        self.update()

    def _start_anim(self, step_snap: int, direction: int) -> None:
        """Animate the step that produces snapshot ``step_snap``; direction -1 reverses."""
        self._anim_to = step_snap
        self._anim_dir = direction
        self._anim_t = 0.0 if direction > 0 else 1.0
        self._anim_start = time.perf_counter()
        if self.animation_ms <= 0:
            self._anim_t = 1.0
            return
        self._timer.start()

    @property
    def animating(self) -> bool:
        return self._timer.isActive() or self._frozen

    def freeze_animation(self, t: float) -> None:
        """Hold the animation of the current snapshot's step at time t (screenshots, tests)."""
        self._timer.stop()
        self._anim_to = self.session.snap_index
        self._anim_t = float(t)
        self._frozen = True
        self.update()

    def unfreeze(self) -> None:
        self._frozen = False
        self.update()

    def _tick(self) -> None:
        el = (time.perf_counter() - self._anim_start) * 1000.0
        t = min(1.0, el / max(1, self.animation_ms))
        self._anim_t = t if self._anim_dir > 0 else 1.0 - t
        if t >= 1.0:
            self._timer.stop()
        self.update()

    def current_frame(self) -> anim.Frame:
        s = self.session
        tr = s.trace
        mode = s.color_mode
        if self.animating and self._anim_to < len(tr):
            snap = tr[self._anim_to]
            prev = tr[max(0, self._anim_to - 1)]
            t = self._anim_t
        else:
            snap = s.snapshot
            prev = s.prev_snapshot
            t = 1.0
        prev_bits = K.lanes_to_bits(prev.state).reshape(-1)
        cur_bits = K.lanes_to_bits(snap.state).reshape(-1)
        diff = None
        if mode == "avalanche":
            av = s.avalanche()
            diff = av.diff_bits(snap.index).reshape(-1)
        return anim.build_frame(snap.step, prev_bits, cur_bits, t, mode, diff, snap.skipped,
                                snap.theta_c, snap.theta_d, self.show_lines)

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

        self._apply_highlights(frame)
        data = frame.instance_data()
        self.inst.write(data.tobytes())
        self.prog["mvp"].write(mvp_bytes)
        self.vao.render(instances=len(data))

        ctx.disable(moderngl.CULL_FACE)
        self.line_prog["mvp"].write(mvp_bytes)
        verts = self._line_verts(frame, overlay=False)
        if len(verts):
            self.line_vbo.write(verts.tobytes())
            self.line_vao.render(moderngl.LINES, vertices=len(verts))
        verts = self._line_verts(frame, overlay=True)
        if len(verts):
            ctx.disable(moderngl.DEPTH_TEST)
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
            # sources for the next step
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

    def _next_step(self) -> Optional[str]:
        s = self.session
        if s.snap_index + 1 < s.num_snapshots:
            return s.trace[s.snap_index + 1].step
        return None

    AXES_ORIGIN = np.array([-3.4, -3.4, 33.4])

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
        else:
            if s.structure:
                blo, bhi = anim.structure_bounds(s.structure, *anchor)
                a, b = anim.box_lines(blo - 0.05, bhi + 0.05)
                segs.append((a, b, anim.STRUCTURE_COLORS[s.structure] + (0.95,)))
            if s.selected is not None and self.show_lines and not self.animating:
                nxt = self._next_step()
                if nxt:
                    a, b, _ = anim.feed_lines(nxt, *s.selected, frame.pos)
                    segs.append((a, b, (0.2, 1.0, 0.9, 0.9)))
                i = anim.cell_index(*s.selected)
                p = frame.pos[i]
                a, b = anim.box_lines(p - 0.55, p + 0.55)
                segs.append((a, b, (1.0, 1.0, 1.0, 1.0)))
            segs.extend(frame.overlay_lines)
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

    # ------------------------------------------------------------ overlay

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
            sp = self.camera.project(mvp, pts, w, h)
            labels = [("x  (row →)", (255, 90, 90)), ("y  (column ↑)", (90, 255, 115)),
                      ("z  (lane, 64 bits)", (100, 150, 255)), ("(0,0,0)", (200, 200, 200))]
            for (sx, sy, wv), (txt, col) in zip(sp, labels):
                if wv > 0:
                    p.setPen(QtGui.QColor(*col))
                    p.drawText(QtCore.QPointF(sx + 4, sy - 4), txt)
        # HUD
        s = self.session
        snap = s.snapshot
        p.setPen(QtGui.QColor(235, 235, 240))
        f2 = QtGui.QFont(font)
        f2.setPointSize(12)
        f2.setBold(True)
        p.setFont(f2)
        p.drawText(12, 22, s.position_text())
        p.setFont(font)
        y = 42
        if snap.step != "initial":
            p.setPen(QtGui.QColor(190, 190, 200))
            p.drawText(12, y, K.step_description(snap.step))
            y += 18
            if snap.step == "iota" and snap.round_constant is not None:
                p.drawText(12, y, f"RC[{snap.round_index_abs}] = 0x{snap.round_constant:016x}")
                y += 18
        if snap.skipped:
            p.setPen(QtGui.QColor(255, 120, 120))
            p.drawText(12, y, "step disabled in parameters - state unchanged")
            y += 18
        # legend
        y += 6
        legend = {
            "raw": [("1", anim.COL_ONE), ("0", anim.COL_ZERO)],
            "changed": [("flipped to 1", anim.COL_CHANGED_TO_ONE), ("flipped to 0", anim.COL_CHANGED_TO_ZERO),
                        ("unchanged 1", anim.COL_UNCHANGED_ONE), ("unchanged 0", anim.COL_ZERO)],
            "avalanche": [("differs from flipped run", anim.COL_DIFF), ("same, value 1", anim.COL_DIFF_DIM_ONE),
                          ("same, value 0", anim.COL_ZERO)],
        }[s.color_mode]
        if snap.step == "theta" and self.animating:
            legend = legend + [("C[x] parity sheet", anim.COL_C), ("D[x] correction sheet", anim.COL_D)]
        for name, col in legend:
            p.fillRect(12, y - 10, 12, 12, QtGui.QColor(*(int(c * 255) for c in col)))
            p.setPen(QtGui.QColor(200, 200, 210))
            p.drawText(30, y, name)
            y += 17
        # selection info
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
        p.setPen(QtGui.QColor(120, 120, 130))
        p.drawText(w - 150, h - 8, f"{self.fps:4.0f} fps  |  1600 cells, 1 draw call")

    # ------------------------------------------------------------ picking

    def pick(self, px: float, py: float) -> Optional[Tuple[int, int, int]]:
        if self._frame is None:
            return None
        mvp = self._mvp()
        o, d = self.camera.ray(mvp, px, py, self.width(), self.height())
        pos = self._frame.pos[:1600].astype(np.float64)
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
        self._dragging = False
        self.setFocus()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        dx = ev.x() - self._last_pos[0]
        dy = ev.y() - self._last_pos[1]
        self._last_pos = (ev.x(), ev.y())
        if ev.buttons() & QtCore.Qt.LeftButton and not (ev.modifiers() & QtCore.Qt.ShiftModifier):
            self.camera.orbit(dx, dy)
            self._dragging = True
            self.update()
        elif ev.buttons() & (QtCore.Qt.RightButton | QtCore.Qt.MiddleButton) or (
                ev.buttons() & QtCore.Qt.LeftButton):
            self.camera.pan(dx, dy, self.height())
            self._dragging = True
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
            cell = self.pick(ev.x(), ev.y())
            self.session.select(cell)
        self._dragging = False

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.session.select(None)

    def leaveEvent(self, ev) -> None:
        self._hover = None
        self.update()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        self.camera.zoom(ev.angleDelta().y() / 120.0)
        self.update()

    def set_preset(self, name: str) -> None:
        self.camera.preset(name)
        self.camera.target = np.array([0.0, 0.0, 5.0]) if name == "perspective" else np.zeros(3)
        self.camera.distance = {"lane-on": 95.0, "top": 95.0, "slice-on": 40.0, "isometric": 80.0}.get(name, 100.0)
        self.update()

    def set_animation_ms(self, ms: int) -> None:
        self.animation_ms = ms

    def grab_png(self, path: str) -> None:
        self.grabFramebuffer().save(path)
