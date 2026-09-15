"""Batched lanes: N independent sponge instances, lane (x,y) of every
instance packed into one vector register."""

from __future__ import annotations

from typing import List

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import keccak as K
from ...core import layouts as L
from ...core import sponge as S
from .base import C_DIM, C_SEL, C_TEXT, MONO, ModeWidget

INST_COLORS = [QtGui.QColor(c) for c in ("#ffbd2e", "#5ec9f5", "#ff6fae", "#8cf07a", "#c99cff", "#ffa46b", "#7de3d3", "#f2f26b")]


class BatchedView(ModeWidget):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self.setFont(MONO)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(12, 50, 12, 0)
        bar.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        bar.addWidget(QtWidgets.QLabel("instances N"))
        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(2, 8)
        self.n_spin.setValue(session.params.batch_n)
        self.n_spin.valueChanged.connect(lambda v: session.set_params(batch_n=v))
        bar.addWidget(self.n_spin)
        bar.addWidget(QtWidgets.QLabel("   instance i hashes the message with \"-i\" appended (instance 0 = the real message)"))
        bar.addStretch(1)
        self.setLayout(bar)
        self._batch_trace = None
        self._batch_key = None

    def _batch(self):
        s = self.session
        p = s.params
        key = (p.batch_n, p.message, p.rate_bytes, p.domain_byte, p.num_rounds, p.enabled_steps, s.perm_index,
               p.round_offset_standard)
        if key != self._batch_key:
            msgs = [p.message] + [p.message + f"-{i}".encode() for i in range(1, p.batch_n)]
            states = []
            for m in msgs:
                run = S.sponge(m, p.variant_obj(), 0, p.num_rounds, p.enabled_steps, trace=False,
                               round_offset=None if p.round_offset_standard else 0)
                # take the same permutation call index if it exists, else the last one
                idx = min(s.perm_index, run.num_perm_calls - 1)
                states.append(run.perm_calls[idx].state_in)
            batch = L.batch_states(states)
            _, tr = K.permute(batch, p.num_rounds, trace=True, enabled_steps=p.enabled_steps,
                              round_offset=None if p.round_offset_standard else 0)
            self._batch_trace = tr
            self._batch_key = key
        return self._batch_trace

    def on_run_changed(self) -> None:
        self._batch_key = None
        self.update()

    def cell_at(self, px, py):
        top, rowh = 100, 20
        lane = (py - top) // rowh
        if 0 <= lane < 25:
            return int(lane) % 5, int(lane) // 5, 0
        return None

    def paint(self, p: QtGui.QPainter) -> None:
        s = self.session
        tr = self._batch()
        snap = tr[min(s.snap_index, len(tr) - 1)]
        n = snap.state.shape[0]
        self.draw_header(p, f"Batched lanes (SIMD layout) — {s.position_text()}",
                         f"Register R[x,y] holds lane (x,y) of all {n} instances. Every step mapping is element-wise across the "
                         "register: element i only ever meets element i of other registers, never its neighbours.")
        f = QtGui.QFont(MONO)
        f.setPointSize(9)
        p.setFont(f)
        top, rowh = 100, 20
        colw = 150
        x0 = 130
        p.setPen(C_DIM)
        p.drawText(12, top - 8, "register")
        for i in range(n):
            p.setPen(INST_COLORS[i])
            p.drawText(x0 + i * colw, top - 8, f"elem {i} = inst {i}")
        sel = s.selected or self._hover
        sel_lane = (sel[0], sel[1]) if sel else None
        for lane in range(25):
            x, y = lane % 5, lane // 5
            yy = top + lane * rowh + 14
            if sel_lane == (x, y):
                p.fillRect(QtCore.QRect(8, yy - 14, x0 + n * colw - 8, rowh), QtGui.QColor(60, 62, 75))
            p.setPen(C_TEXT)
            p.drawText(12, yy, f"R[{x},{y}]  ")
            for i in range(n):
                p.setPen(INST_COLORS[i].darker(115))
                p.drawText(x0 + i * colw, yy, f"{int(snap.state[i, x, y]):016x}")
        # ---- theta parity demo
        by = top + 25 * rowh + 24
        sx = sel_lane[0] if sel_lane else 0
        c = L.theta_parity_registers(snap.state)  # (N, 5)
        p.setPen(C_TEXT)
        f.setPointSize(10)
        p.setFont(f)
        p.drawText(12, by, f"θ column parity for sheet x = {sx}:   C[{sx}] = R[{sx},0] ^ R[{sx},1] ^ R[{sx},2] ^ R[{sx},3] ^ R[{sx},4]")
        f.setPointSize(9)
        p.setFont(f)
        yy = by + 22
        for yv in range(5):
            p.setPen(C_DIM)
            p.drawText(12, yy, f"R[{sx},{yv}]")
            for i in range(n):
                p.setPen(INST_COLORS[i].darker(115))
                p.drawText(x0 + i * colw, yy, f"{int(snap.state[i, sx, yv]):016x}")
            yy += rowh
        p.setPen(C_DIM)
        p.drawText(12, yy, "XOR ↓ (element-wise, no cross-element reduction)")
        yy += rowh
        p.setPen(C_TEXT)
        p.drawText(12, yy, f"C[{sx}]")
        for i in range(n):
            p.setPen(INST_COLORS[i])
            p.drawText(x0 + i * colw, yy, f"{int(c[i, sx]):016x}")
        yy += rowh + 6
        ok = L.theta_parity_is_elementwise(snap.state)
        p.setPen(QtGui.QColor(120, 255, 140) if ok else QtGui.QColor(255, 100, 100))
        p.drawText(12, yy, f"each C[{sx}] element equals the parity of its own instance: {'yes' if ok else 'NO'}.  "
                           "ρ is a per-element rotate, π is register renaming, χ is register-wide AND/NOT/XOR, ι a broadcast XOR.")
        p.setPen(C_DIM)
        p.drawText(12, yy + 18, "Contrast with the NTT of ML-DSA, where butterflies must cross lanes: Keccak needs no shuffles at all "
                                "once instances are packed this way (N-way parallel hashing, e.g. for SHAKE in signing).")
