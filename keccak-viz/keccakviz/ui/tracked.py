"""Dock panel listing the tracked bits and their per-step history."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ..core import keccak as K
from .session import Session


class TrackedPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("track selected cell / structure  (F)")
        self.add_btn.setToolTip("Click a cell in any view, then press F or this button to follow that bit.  "
                                "With a substructure chip active (row, column, lane, …) the whole structure is tracked.")
        self.add_btn.clicked.connect(lambda: session.track_focus(session.selected))
        row.addWidget(self.add_btn)
        self.clear_btn = QtWidgets.QPushButton("clear")
        self.clear_btn.clicked.connect(session.clear_tracked)
        row.addWidget(self.clear_btn)
        lay.addLayout(row)
        self.hint = QtWidgets.QLabel("A tracked bit keeps its marker as ρ and π move it and records every flip. "
                                     "Its trail is drawn in the 3D view.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #9aa;")
        lay.addWidget(self.hint)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(140)
        self.list.itemDoubleClicked.connect(self._remove_item)
        self.list.setToolTip("double-click an entry to stop tracking it")
        lay.addWidget(self.list)
        self.history = QtWidgets.QTextBrowser()
        lay.addWidget(self.history, 1)
        session.trackedChanged.connect(self.refresh)
        session.positionChanged.connect(lambda *_: self.refresh())
        session.runChanged.connect(self.refresh)
        session.selectionChanged.connect(lambda c: self.add_btn.setEnabled(c is not None))
        self.add_btn.setEnabled(session.selected is not None)
        self.refresh()

    def _remove_item(self, item: QtWidgets.QListWidgetItem) -> None:
        self.session.untrack(int(item.data(QtCore.Qt.UserRole)))

    def refresh(self) -> None:
        s = self.session
        k = s.snap_index
        self.list.clear()
        html = ["<style>body{color:#e1e1e6;font-size:9pt} td{padding:1px 6px}</style>"]
        for ci, group in enumerate(s.tracked):
            col = s.TRACK_COLORS[ci]
            tracks = [s.bit_track(o) for o in group.origins]
            if group.single:
                track = tracks[0]
                pos = track.position(k)
                text = (f"#{ci + 1}  {group.label}  now at ({pos[0]},{pos[1]},{pos[2]}) = {track.value(k)}   "
                        f"flips so far: {sum(1 for e in track.events[:k + 1] if e.startswith('flipped'))}")
            else:
                ones = sum(t.value(k) for t in tracks)
                text = f"#{ci + 1}  {group.label}: {len(tracks)} bits, {ones} of them 1 now"
            item = QtWidgets.QListWidgetItem(text)
            item.setForeground(QtGui.QColor(col))
            item.setData(QtCore.Qt.UserRole, ci)
            self.list.addItem(item)
            html.append(f"<p style='color:{col}'><b>#{ci + 1}</b> {group.label}</p>")
            html.append("<table>")
            lo, hi = max(0, k - 6), min(len(tracks[0].events) - 1, k + 3)
            for j in range(lo, hi + 1):
                snap = s.trace[j]
                label = "initial" if snap.step == "initial" else f"r{snap.round + 1} {K.STEP_SYMBOLS[snap.step]}"
                style = " style='background:#3a3f55'" if j == k else ""
                if group.single:
                    t = tracks[0]
                    p = t.position(j)
                    html.append(f"<tr{style}><td>{label}</td><td>({p[0]},{p[1]},{p[2]})</td>"
                                f"<td>{t.value(j)}</td><td>{t.events[j]}</td></tr>")
                else:
                    moved = sum(1 for t in tracks if t.events[j].startswith("moved"))
                    flipped = sum(1 for t in tracks if t.events[j].startswith("flipped"))
                    ones = sum(t.value(j) for t in tracks)
                    where = sorted({t.position(j)[2] for t in tracks})
                    zs = f"z∈{{{where[0]}…{where[-1]}}}" if len(where) > 4 else "z=" + ",".join(map(str, where))
                    html.append(f"<tr{style}><td>{label}</td><td>{ones} ones</td>"
                                f"<td>{moved} moved</td><td>{flipped} flipped, {zs}</td></tr>")
            html.append("</table>")
        if not s.tracked:
            html.append("<p style='color:#9aa'>Nothing tracked yet.  Click a cell, then press F.  With a "
                        "substructure chip active, F tracks the whole row / column / lane / slice / plane / sheet.</p>")
        self.history.setHtml("".join(html))
