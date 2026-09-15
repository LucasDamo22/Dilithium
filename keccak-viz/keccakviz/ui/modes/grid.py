"""A 25 x 64 'unrolled state' grid (rows = lanes, columns = z) used by the
diffusion heatmap and the avalanche view."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from PyQt5 import QtCore, QtGui

from .base import C_DIM, C_SEL, C_TEXT


def draw_lane_grid(p: QtGui.QPainter, x0: int, y0: int, cw: float, ch: float,
                   color_of: Callable[[int, int, int], QtGui.QColor],
                   selected: Optional[Tuple[int, int, int]] = None,
                   hover: Optional[Tuple[int, int, int]] = None) -> None:
    """Rows are lanes in order x + 5y (0..24), columns are z (0..63)."""
    p.setPen(C_DIM)
    for lane in range(25):
        x, y = lane % 5, lane // 5
        yy = y0 + lane * ch
        p.drawText(QtCore.QPointF(x0 - 84, yy + ch * 0.75), f"{lane:2d} A[{x},{y}]")
        if lane % 5 == 0 and lane:
            p.setPen(QtGui.QColor(90, 90, 100))
            p.drawLine(QtCore.QPointF(x0, yy), QtCore.QPointF(x0 + 64 * cw, yy))
            p.setPen(C_DIM)
        for z in range(64):
            r = QtCore.QRectF(x0 + z * cw, yy, cw, ch)
            p.fillRect(r.adjusted(0.5, 0.5, -0.5, -0.5), color_of(x, y, z))
    p.setPen(C_DIM)
    for z in range(0, 64, 8):
        p.drawText(QtCore.QPointF(x0 + z * cw + 2, y0 - 4), f"z={z}")
    p.drawText(QtCore.QPointF(x0 + 64 * cw + 6, y0 + 12 * ch), "lane index = x + 5y")
    for cell, pen in ((hover, QtGui.QPen(C_TEXT, 1)), (selected, QtGui.QPen(C_SEL, 2))):
        if cell:
            x, y, z = cell
            p.setPen(pen)
            p.drawRect(QtCore.QRectF(x0 + z * cw, y0 + (x + 5 * y) * ch, cw, ch))


def grid_cell_at(px: int, py: int, x0: int, y0: int, cw: float, ch: float):
    z = int((px - x0) // cw)
    lane = int((py - y0) // ch)
    if 0 <= z < 64 and 0 <= lane < 25:
        return lane % 5, lane // 5, z
    return None


def grid_geometry(width: int, height: int, top: int, bottom: int) -> Tuple[int, int, float, float]:
    x0 = 110
    cw = max(4.0, min(18.0, (width - x0 - 160) / 64))
    ch = max(4.0, min(18.0, (height - top - bottom) / 25))
    return x0, top, cw, ch


def viridis(t: float) -> QtGui.QColor:
    """Cheap viridis-like ramp for t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return QtGui.QColor(*(int(a + (b - a) * u) for a, b in zip(c0, c1)))
    return QtGui.QColor(*stops[-1][1])
