"""Main window and entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from .session import Params, Session


def _make_app(argv) -> QtWidgets.QApplication:
    from .glview import install_surface_format

    install_surface_format()
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("Keccak visualizer")
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(37, 39, 46))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(225, 225, 230))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(28, 30, 36))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(44, 46, 54))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(225, 225, 230))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(50, 52, 60))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(225, 225, 230))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(70, 120, 200))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(50, 52, 60))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(225, 225, 230))
    pal.setColor(QtGui.QPalette.Link, QtGui.QColor(120, 170, 255))
    app.setPalette(pal)
    return app


class StructureLegend(QtWidgets.QWidget):
    """Row of hoverable / checkable chips for row, column, lane, slice, plane, sheet."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        from . import anim

        self.session = session
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.addWidget(QtWidgets.QLabel("substructures:"))
        self.buttons: Dict[str, QtWidgets.QToolButton] = {}
        tips = {
            "row": "row: 5 bits along x, fixed (y, z) - chi works on rows",
            "column": "column: 5 bits along y, fixed (x, z) - theta sums columns",
            "lane": "lane: 64 bits along z, fixed (x, y) - rho rotates lanes, pi moves them",
            "slice": "slice: 25 bits, fixed z - theta and chi stay inside a slice",
            "plane": "plane: 5 lanes with the same y (320 bits)",
            "sheet": "sheet: 5 lanes with the same x (320 bits) - a theta column parity C[x] is a sheet parity",
        }
        for name in session.STRUCTURES:
            b = QtWidgets.QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setToolTip(tips[name])
            r, g, bl = (int(c * 255) for c in anim.STRUCTURE_COLORS[name])
            b.setStyleSheet(f"QToolButton{{color: rgb({r},{g},{bl}); font-weight: bold;}}"
                            f"QToolButton:checked{{background: rgba({r},{g},{bl},60);}}")
            b.installEventFilter(self)
            b.clicked.connect(lambda checked, n=name: self._clicked(n, checked))
            lay.addWidget(b)
            self.buttons[name] = b
        lay.addStretch(1)
        self._checked: Optional[str] = None

    def eventFilter(self, obj, ev):
        for name, b in self.buttons.items():
            if obj is b:
                if ev.type() == QtCore.QEvent.Enter:
                    self.session.set_structure(name)
                elif ev.type() == QtCore.QEvent.Leave:
                    self.session.set_structure(self._checked)
        return super().eventFilter(obj, ev)

    def _clicked(self, name: str, checked: bool) -> None:
        self._checked = name if checked else None
        for n, b in self.buttons.items():
            b.setChecked(n == self._checked)
        self.session.set_structure(self._checked)



class CubeMode(QtWidgets.QWidget):
    """The 3D cube plus its legend and camera controls."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        from .glview import CubeView

        self.session = session
        self.view = CubeView(session)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(6, 2, 6, 2)
        top.addWidget(QtWidgets.QLabel("camera:"))
        for name in ("perspective", "isometric", "slice-on", "lane-on", "top"):
            b = QtWidgets.QToolButton()
            b.setText(name)
            b.setAutoRaise(True)
            b.clicked.connect(lambda _c, n=name: self.view.set_preset(n))
            top.addWidget(b)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("colour:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(["raw 0/1", "changed since previous step", "avalanche difference"])
        self.color_combo.currentIndexChanged.connect(
            lambda i: session.set_color_mode(session.COLOR_MODES[i]))
        top.addWidget(self.color_combo)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("lines:"))
        self.lines_combo = QtWidgets.QComboBox()
        self.lines_combo.addItems(["none", "focused (selected + tracked bits)", "all"])
        self.lines_combo.setCurrentIndex(1)
        self.lines_combo.setToolTip("How many feed lines / arrows to draw during step animations")
        self.lines_combo.currentIndexChanged.connect(lambda i: self.view.set_line_mode(self.view.LINE_MODES[i]))
        top.addWidget(self.lines_combo)
        self.labels_cb = QtWidgets.QCheckBox("labels")
        self.labels_cb.setChecked(True)
        self.labels_cb.toggled.connect(self._toggle_labels)
        top.addWidget(self.labels_cb)
        top.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addWidget(StructureLegend(session))
        session.colorModeChanged.connect(self._sync_color)

    def _toggle_labels(self, on: bool) -> None:
        self.view.show_labels = on
        self.view.update()

    def _sync_color(self, mode: str) -> None:
        i = self.session.COLOR_MODES.index(mode)
        if self.color_combo.currentIndex() != i:
            self.color_combo.setCurrentIndex(i)


class MainWindow(QtWidgets.QMainWindow):
    MODES = [
        ("cube", "3D cube"),
        ("slices", "Slice stack"),
        ("lanes", "Lane table"),
        ("bytes", "State bytes"),
        ("heatmap", "Diffusion heatmap"),
        ("avalanche", "Avalanche"),
        ("interleaved", "Bit-interleaved"),
        ("batched", "Batched lanes"),
        ("sponge", "Sponge"),
    ]

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.setWindowTitle("Keccak-f[1600] / SHA-3 visualizer")
        self.resize(1500, 950)

        from .transport import Transport

        self.transport = Transport(session)
        self.stack = QtWidgets.QStackedWidget()
        self.modes: Dict[str, QtWidgets.QWidget] = {}
        self.cube = CubeMode(session)
        self._add_mode("cube", self.cube)
        self.transport.speedChanged.connect(self.cube.view.set_animation_ms)
        self.cube.view.set_animation_ms(self.transport.anim_ms)
        self.transport.scrubbed.connect(self.cube.view.freeze_animation)
        self.cube.view.animProgress.connect(self.transport.set_progress)
        self._build_other_modes()

        self.mode_tabs = QtWidgets.QTabBar()
        self.mode_tabs.setExpanding(False)
        for key, title in self.MODES:
            self.mode_tabs.addTab(title)
        self.mode_tabs.currentChanged.connect(self._on_tab)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.mode_tabs)
        lay.addWidget(self.stack, 1)
        lay.addWidget(self.transport)
        self.setCentralWidget(central)

        session.jump_to_cube.connect(lambda: self.set_mode("cube"))
        self._build_docks()
        self._build_menu()
        self._build_shortcuts()
        self.statusBar().showMessage("Left-drag orbit · right/middle-drag or shift-drag pan · wheel zoom · click a cell · "
                                     "←/→ step · ↑/↓ round · space play")
        self.cube.view.fpsMeasured.connect(lambda f: self.status_fps.setText(f"{f:.0f} fps"))
        self.status_fps = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self.status_fps)

    # ------------------------------------------------------------ modes

    def _add_mode(self, key: str, widget: QtWidgets.QWidget) -> None:
        self.modes[key] = widget
        self.stack.addWidget(widget)

    def _build_other_modes(self) -> None:
        from .modes import make_modes

        for key, widget in make_modes(self.session):
            self._add_mode(key, widget)

    def _on_tab(self, i: int) -> None:
        key = self.MODES[i][0]
        w = self.modes.get(key)
        if w is not None:
            self.stack.setCurrentWidget(w)
            if hasattr(w, "activated"):
                w.activated()
        if hasattr(self, "explain"):
            self.explain.set_mode(key)

    def set_mode(self, key: str) -> None:
        for i, (k, _t) in enumerate(self.MODES):
            if k == key:
                self.mode_tabs.setCurrentIndex(i)
                self._on_tab(i)
                return

    @property
    def current_mode(self) -> str:
        return self.MODES[self.mode_tabs.currentIndex()][0]

    # ------------------------------------------------------------ docks

    def _build_docks(self) -> None:
        from .params import ParamsPanel
        from .explain import ExplainPanel

        self.params_panel = ParamsPanel(self.session)
        d1 = QtWidgets.QDockWidget("Parameters", self)
        d1.setObjectName("params")
        d1.setWidget(self.params_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d1)

        self.explain = ExplainPanel(self.session)
        d2 = QtWidgets.QDockWidget("Explanation", self)
        d2.setObjectName("explain")
        d2.setWidget(self.explain)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d2)

        from .tracked import TrackedPanel

        self.tracked_panel = TrackedPanel(self.session)
        d3 = QtWidgets.QDockWidget("Tracked bits", self)
        d3.setObjectName("tracked")
        d3.setWidget(self.tracked_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d3)
        self.tabifyDockWidget(d2, d3)
        d2.raise_()
        self.resizeDocks([d1, d2], [380, 520], QtCore.Qt.Vertical)
        self.resizeDocks([d1, d2, d3], [400, 400, 400], QtCore.Qt.Horizontal)
        self.explain.set_mode("cube")

    # ------------------------------------------------------------ menu / keys

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        a = m.addAction("Save view as PNG…")
        a.setShortcut("Ctrl+S")
        a.triggered.connect(self.export_png)
        a = m.addAction("Export trace as JSON…")
        a.setShortcut("Ctrl+E")
        a.triggered.connect(self.export_json)
        m.addSeparator()
        a = m.addAction("Quit")
        a.setShortcut("Ctrl+Q")
        a.triggered.connect(self.close)
        v = self.menuBar().addMenu("&View")
        for i, (key, title) in enumerate(self.MODES):
            a = v.addAction(f"&{i + 1}  {title}")
            a.setShortcut(str(i + 1))
            a.triggered.connect(lambda _c, k=key: self.set_mode(k))
        v.addSeparator()
        for name, key in (("perspective", "V"), ("isometric", "I"), ("slice-on", "S"), ("lane-on", "L"), ("top", "T")):
            a = v.addAction(f"camera: {name}")
            a.setShortcut(key)
            a.triggered.connect(lambda _c, n=name: self.cube.view.set_preset(n))
        v.addSeparator()
        for i, (mode, title) in enumerate(zip(self.session.COLOR_MODES, ("raw 0/1", "changed", "avalanche"))):
            a = v.addAction(f"colour: {title}")
            a.setShortcut(f"Ctrl+{i + 1}")
            a.triggered.connect(lambda _c, mm=mode: self.session.set_color_mode(mm))
        h = self.menuBar().addMenu("&Help")
        a = h.addAction("Keyboard shortcuts")
        a.triggered.connect(self.show_help)

    def _build_shortcuts(self) -> None:
        s = self.session
        t = self.transport

        def sc(keys, slot):
            for k in keys:
                a = QtWidgets.QShortcut(QtGui.QKeySequence(k), self)
                a.setContext(QtCore.Qt.ApplicationShortcut)
                a.activated.connect(slot)

        sc(["Right", "."], s.step_forward)
        sc(["Left", ","], s.step_back)
        sc(["Up", "PgDown", "]"], s.round_forward)
        sc(["Down", "PgUp", "["], s.round_back)
        sc(["Home"], s.go_start)
        sc(["End"], s.go_end)
        sc(["Space", "P"], t.toggle_play)
        sc(["Escape"], lambda: s.select(None))
        sc(["F"], lambda: s.track(s.selected))
        sc(["Shift+F"], s.clear_tracked)
        sc(["N"], lambda: s.set_position(perm_index=s.perm_index + 1, snap_index=0))
        sc(["B"], lambda: s.set_position(perm_index=s.perm_index - 1, snap_index=0))

    def show_help(self) -> None:
        QtWidgets.QMessageBox.information(self, "Keyboard shortcuts", HELP_TEXT)

    # ------------------------------------------------------------ export

    def export_png(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PNG", "keccak.png", "PNG (*.png)")
        if path:
            self.save_png(path)

    def save_png(self, path: str) -> None:
        w = self.stack.currentWidget()
        if w is self.cube:
            img = self.cube.view.grabFramebuffer()
        else:
            img = w.grab().toImage()
        img.save(path)
        self.statusBar().showMessage(f"saved {path}", 4000)

    def export_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export trace JSON", "keccak_trace.json", "JSON (*.json)")
        if path:
            self.save_json(path)

    def save_json(self, path: str) -> None:
        from ..core.export import run_to_json

        with open(path, "w") as f:
            f.write(run_to_json(self.session.run))
        self.statusBar().showMessage(f"exported {path}", 4000)


HELP_TEXT = """
Navigation
  →  or .        next step mapping          ←  or ,   previous step mapping
  ↑  or ]        next round                 ↓  or [   previous round
  Home / End     initial / final state      Space, P  play / pause
  N / B          next / previous permutation call (multi-block messages, long SHAKE output)
  1 … 9          switch visualization mode  Esc       clear selection
  F / Shift+F    track the selected bit / clear tracked bits
  step scrubber  drag to animate the current step by hand; drag into an end zone to change step

Camera (3D cube)
  left-drag      orbit                      right- or middle-drag, shift-drag   pan
  wheel          zoom                       V / I / S / L / T   perspective / isometric / slice-on / lane-on / top
  click a cell   show its coordinates, lane, value and what feeds it next
  Ctrl+1/2/3     colour mode raw / changed / avalanche

File
  Ctrl+S   save the current view as PNG     Ctrl+E   export the whole sponge run as JSON
"""


def parse_args(argv):
    p = argparse.ArgumentParser(prog="keccakviz", description="Interactive Keccak / SHA-3 visualizer")
    p.add_argument("-m", "--message", default="abc", help="input message (text)")
    p.add_argument("--hex", action="store_true", help="interpret --message as hex")
    p.add_argument("--variant", default="SHA3-256", help="SHA3-224/256/384/512, SHAKE128/256, Keccak-224/256/384/512")
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--mode", default="cube")
    p.add_argument("--snap", type=int, default=0, help="initial snapshot index")
    p.add_argument("--perm", type=int, default=0, help="initial permutation call index")
    p.add_argument("--preset", default=None, help="camera preset")
    p.add_argument("--color", default=None, choices=Session.COLOR_MODES)
    p.add_argument("--select", default=None, help="x,y,z cell to select")
    p.add_argument("--structure", default=None, choices=Session.STRUCTURES)
    p.add_argument("--anim-t", type=float, default=None,
                   help="freeze the step animation at this t (0..1), for screenshots")
    p.add_argument("--screenshot", default=None, help="save a PNG of the initial view and exit")
    p.add_argument("--screenshot-window", default=None, help="save a PNG of the whole window and exit")
    p.add_argument("--json", default=None, help="export the run as JSON and exit")
    p.add_argument("--detail", type=int, default=None, help="explanation detail 0/1/2")
    p.add_argument("--track", default=None, help="comma-separated x,y,z triples to track, e.g. 0,0,0;1,2,3")
    p.add_argument("--bench", type=float, default=None,
                   help="play the 3D animation for this many seconds, print the frame rate, and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from ..core import sponge as S

    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.variant not in S.VARIANTS:
        print("unknown variant; choose from", ", ".join(S.VARIANTS), file=sys.stderr)
        return 2
    msg = bytes.fromhex(args.message) if args.hex else args.message.encode()
    params = Params(message=msg, message_is_hex=args.hex, num_rounds=args.rounds)
    v = S.VARIANTS[args.variant]
    params.variant, params.rate_bytes, params.domain_byte = v.name, v.rate_bytes, v.domain_byte
    params.output_bytes = v.output_bytes or 32

    if args.json and not args.screenshot:
        from ..core.export import run_to_json

        session = Session(params)  # QObject without an app is fine for signals
        with open(args.json, "w") as f:
            f.write(run_to_json(session.run))
        return 0

    app = _make_app([sys.argv[0]])
    session = Session(params)
    win = MainWindow(session)
    win.show()
    win.set_mode(args.mode)
    session.set_position(perm_index=args.perm, snap_index=args.snap)
    if args.preset:
        win.cube.view.set_preset(args.preset)
    if args.color:
        session.set_color_mode(args.color)
    if args.select:
        session.select(tuple(int(v) for v in args.select.split(",")))
    if args.structure:
        session.set_structure(args.structure)
    if args.detail is not None:
        session.set_detail_level(args.detail)
    if args.track:
        for trip in args.track.split(";"):
            session.tracked.append(tuple(int(v) for v in trip.split(",")))
        session.trackedChanged.emit()
    if args.anim_t is not None:
        win.cube.view.freeze_animation(args.anim_t)

    if args.bench:
        samples = []
        win.cube.view.fpsMeasured.connect(samples.append)
        win.transport.speed.setCurrentIndex(1)  # fast: continuous stepping
        win.transport.toggle_play()

        def done():
            win.transport.stop()
            fps = sum(samples) / max(1, len(samples))
            print(f"bench: {len(samples)} s, mean {fps:.1f} fps while animating, "
                  f"reached snapshot {session.snap_index}")
            app.quit()
        QtCore.QTimer.singleShot(int(args.bench * 1000), done)
        return app.exec_()

    if args.screenshot or args.screenshot_window:
        def shot():
            if args.screenshot:
                win.save_png(args.screenshot)
            if args.screenshot_window:
                win.grab().save(args.screenshot_window)
            if args.json:
                win.save_json(args.json)
            app.quit()
        QtCore.QTimer.singleShot(900, shot)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
