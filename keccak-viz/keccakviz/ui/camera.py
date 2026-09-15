"""Orbit camera with pan and zoom, plus named presets.  Pure NumPy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def orthographic(half_h: float, aspect: float, near: float, far: float) -> np.ndarray:
    half_w = half_h * aspect
    m = np.eye(4)
    m[0, 0] = 1 / half_w
    m[1, 1] = 1 / half_h
    m[2, 2] = -2 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


PRESETS = {
    # name: (yaw deg, pitch deg, orthographic?)
    "perspective": (-64.0, 22.0, False),  # free 3/4 view (default)
    "isometric": (-45.0, 35.264, True),  # classic isometric: all three axes foreshortened equally
    "slice-on": (0.0, 0.0, True),  # looking along z: see an x-y slice
    "lane-on": (90.0, 0.0, True),  # looking along x: see the y-z sheet, lanes run left-right
    "top": (0.0, 89.0, True),  # looking down y: see an x-z plane
}


@dataclass
class OrbitCamera:
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = -64.0  # degrees around world Y
    pitch: float = 22.0  # degrees above the x-z plane
    distance: float = 90.0
    fov: float = 30.0
    ortho: bool = False

    def eye(self) -> np.ndarray:
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        d = np.array([sy * cp, sp, cy * cp])
        return self.target + d * self.distance

    def view(self) -> np.ndarray:
        return look_at(self.eye(), self.target, np.array([0.0, 1.0, 0.0]))

    def proj(self, aspect: float) -> np.ndarray:
        near, far = 0.5, self.distance * 4 + 200
        if self.ortho:
            return orthographic(self.distance * math.tan(math.radians(self.fov) / 2), aspect, -far, far)
        return perspective(self.fov, aspect, near, far)

    def mvp(self, aspect: float) -> np.ndarray:
        return self.proj(aspect) @ self.view()

    # ------------------------------------------------------------ interaction

    def orbit(self, dx_px: float, dy_px: float) -> None:
        self.yaw = (self.yaw - dx_px * 0.4) % 360
        self.pitch = max(-89.0, min(89.0, self.pitch + dy_px * 0.4))

    def pan(self, dx_px: float, dy_px: float, height_px: float) -> None:
        # move the target in the screen plane, scaled so a drag follows the cursor
        v = self.view()
        right = v[0, :3]
        up = v[1, :3]
        scale = 2 * self.distance * math.tan(math.radians(self.fov) / 2) / max(1.0, height_px)
        self.target = self.target - right * dx_px * scale + up * dy_px * scale

    def zoom(self, steps: float) -> None:
        self.distance = max(3.0, min(600.0, self.distance * (0.9 ** steps)))

    def preset(self, name: str) -> None:
        self.yaw, self.pitch, self.ortho = PRESETS[name]

    def project(self, mvp: np.ndarray, points: np.ndarray, w: int, h: int) -> np.ndarray:
        """World points (N,3) -> pixel coords (N,3): x, y, and clip-w sign for culling."""
        p = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ mvp.T
        wv = p[:, 3:4]
        ndc = p[:, :3] / np.where(np.abs(wv) < 1e-9, 1e-9, wv)
        sx = (ndc[:, 0] + 1) * 0.5 * w
        sy = (1 - ndc[:, 1]) * 0.5 * h
        return np.stack([sx, sy, wv[:, 0]], axis=1)

    def ray(self, mvp: np.ndarray, px: float, py: float, w: int, h: int):
        """Pixel -> (origin, direction) in world space."""
        inv = np.linalg.inv(mvp)
        nx = 2 * px / max(1, w) - 1
        ny = 1 - 2 * py / max(1, h)
        p0 = inv @ np.array([nx, ny, -1.0, 1.0])
        p1 = inv @ np.array([nx, ny, 1.0, 1.0])
        p0 = p0[:3] / p0[3]
        p1 = p1[:3] / p1[3]
        d = p1 - p0
        return p0, d / np.linalg.norm(d)
