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
        self.add_btn = QtWidgets.QPushButton("track selected cell  (F)")
        self.add_btn.setToolTip("Click a cell in any view, then press F or this button to follow that bit")
        self.add_btn.clicked.connect(lambda: session.track(session.selected))
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
        self.session.untrack(tuple(item.data(QtCore.Qt.UserRole)))

    def refresh(self) -> None:
        s = self.session
        k = s.snap_index
        self.list.clear()
        html = ["<style>body{color:#e1e1e6;font-size:9pt} td{padding:1px 6px}</style>"]
        for ci, origin, track in s.tracked_tracks():
            col = s.TRACK_COLORS[ci]
            pos = track.position(k)
            item = QtWidgets.QListWidgetItem(
                f"#{ci + 1}  started at ({origin[0]},{origin[1]},{origin[2]})  now at "
                f"({pos[0]},{pos[1]},{pos[2]}) = {track.value(k)}   flips so far: "
                f"{sum(1 for e in track.events[:k + 1] if e.startswith('flipped'))}")
            item.setForeground(QtGui.QColor(col))
            item.setData(QtCore.Qt.UserRole, list(origin))
            self.list.addItem(item)
            html.append(f"<p style='color:{col}'><b>#{ci + 1}</b> from ({origin[0]},{origin[1]},{origin[2]})</p>")
            html.append("<table>")
            lo, hi = max(0, k - 6), min(len(track.events) - 1, k + 3)
            for j in range(lo, hi + 1):
                snap = s.trace[j]
                label = "initial" if snap.step == "initial" else f"r{snap.round + 1} {K.STEP_SYMBOLS[snap.step]}"
                p = track.position(j)
                style = " style='background:#3a3f55'" if j == k else ""
                html.append(f"<tr{style}><td>{label}</td><td>({p[0]},{p[1]},{p[2]})</td>"
                            f"<td>{track.value(j)}</td><td>{track.events[j]}</td></tr>")
            html.append("</table>")
        if not s.tracked:
            html.append("<p style='color:#9aa'>Nothing tracked yet.  Click a cell, then press F.</p>")
        self.history.setHtml("".join(html))
