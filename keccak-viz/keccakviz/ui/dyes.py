"""Dock panel for dyes: colour a bit (or a substructure) and watch the colour
spread along the dependency pattern of the step mappings."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from .session import Session


class DyePanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("dye selected cell / structure  (D)")
        self.add_btn.setToolTip("Give the selected bit (or the whole active substructure) a colour")
        self.add_btn.clicked.connect(self._add)
        row.addWidget(self.add_btn)
        self.color_btn = QtWidgets.QPushButton("pick colour…")
        self.color_btn.clicked.connect(self._pick_new_color)
        row.addWidget(self.color_btn)
        self.clear_btn = QtWidgets.QPushButton("clear")
        self.clear_btn.clicked.connect(session.clear_dyes)
        row.addWidget(self.clear_btn)
        lay.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("dye input block (rate)")
        b.setToolTip("Dye every rate bit - the bits the incoming block is XORed into - at this call's permutation input")
        b.clicked.connect(lambda: session.add_region_dye("input"))
        row2.addWidget(b)
        b = QtWidgets.QPushButton("dye capacity")
        b.setToolTip("Dye every capacity bit at this call's permutation input")
        b.clicked.connect(lambda: session.add_region_dye("capacity"))
        row2.addWidget(b)
        b = QtWidgets.QPushButton("both")
        b.setToolTip("Input red, capacity blue: watch them blend")
        b.clicked.connect(lambda: (session.add_region_dye("input"), session.add_region_dye("capacity")))
        row2.addWidget(b)
        lay.addLayout(row2)
        self.next_color = None
        hint = QtWidgets.QLabel(
            "A dye starts on the selected bits at the current frame.  From there every step mixes a bit's dye "
            "into every bit it feeds (an output bit gets the average of its "
            "sources): θ spreads it to 11 cells, χ to 3, ρ and π just carry it.  The total amount of each dye "
            "is conserved, so after enough rounds every cell holds the same mixture.  Brightness = concentration "
            "relative to the strongest cell; hue = the mix of dyes.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa;")
        lay.addWidget(hint)
        self.list = QtWidgets.QListWidget()
        self.list.setToolTip("double-click to change a dye's colour; Delete key removes it")
        self.list.itemDoubleClicked.connect(self._recolor)
        lay.addWidget(self.list, 1)
        self.spread = QtWidgets.QLabel("")
        self.spread.setWordWrap(True)
        self.spread.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(self.spread)
        session.dyesChanged.connect(self.refresh)
        session.positionChanged.connect(lambda *_: self.refresh())
        session.runChanged.connect(self.refresh)
        self.refresh()

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Delete and self.list.currentRow() >= 0:
            self.session.remove_dye(self.list.currentRow())
        else:
            super().keyPressEvent(ev)

    def _add(self) -> None:
        self.session.add_dye(self.session.selected, self.next_color)
        self.next_color = None
        self.color_btn.setText("pick colour…")
        self.color_btn.setStyleSheet("")

    def _pick_new_color(self) -> None:
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor("#ff3b3b"), self, "colour for the next dye")
        if c.isValid():
            self.next_color = c.name()
            self.color_btn.setText(f"next: {c.name()}")
            self.color_btn.setStyleSheet(f"background: {c.name()}; color: black;")

    def _recolor(self, item: QtWidgets.QListWidgetItem) -> None:
        i = self.list.row(item)
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.session.dyes[i].color), self, "dye colour")
        if c.isValid():
            self.session.set_dye_color(i, c.name())

    def refresh(self) -> None:
        s = self.session
        self.list.clear()
        for i, d in enumerate(s.dyes):
            if d.start_load >= 0:
                when = f"perm {d.start_perm + 1}, loading frame {d.start_load}"
            elif d.start_core == 0:
                when = f"perm {d.start_perm + 1}, permutation input"
            else:
                r, st = divmod(d.start_core - 1, 5)
                when = f"perm {d.start_perm + 1}, round {r + 1} after {('θ', 'ρ', 'π', 'χ', 'ι')[st]}"
            item = QtWidgets.QListWidgetItem(f"dye {i + 1}: {d.label}  ({len(d.cells)} bit(s)), from {when}  {d.color}")
            item.setForeground(QtGui.QColor(d.color))
            self.list.addItem(item)
        spread = s.dye_spread()
        if not s.dyes:
            self.spread.setText("No dye yet.  Click a cell (optionally with a substructure chip active) and press D, "
                                "or dye the input block and the capacity with the buttons above.")
            return
        if spread is None:
            self.spread.setText("No dye applied yet at this frame: dyes appear from the frame where they were applied.")
            return
        k = s.snap_index
        arr = s.dye_array()
        reached = int((arr[min(k, len(arr) - 1)].sum(axis=1) > 0).sum())
        lines = [f"<b>Now</b>: cells reached {reached} / 1600;  unevenness max/mean = {spread:.2f} (1.00 = homogeneous)"]
        blend = s.dye_blend()
        if blend is not None:
            lines.append(f"<b>Mixture</b>: {100 * blend:.1f}% unmixed (100% = some cell holds one dye only, "
                         f"0% = every cell holds the same mix)")
            bf = s.blend_frame(0.05)
            lines.append("&nbsp;&nbsp;within 5% everywhere from: " + (s.frame_label(bf) if bf is not None else "not in this call"))
        for i, d in enumerate(s.dyes):
            f = s.dye_full_dependency(i)
            if d.start_perm == s.perm_index:
                when = s.frame_label(f) if f is not None else "not in this call"
                lines.append(f"<b>Dependency</b>, dye {i + 1}: every one of the 1600 bits depends on every bit of it "
                             f"from: {when}")
        lines.append("<span style='color:#9aa'>Mixture = the averaging dye model (proportions).  Dependency = exact: "
                     "the first frame at which no output bit is independent of any dyed bit.</span>")
        self.spread.setText("<br>".join(lines))
