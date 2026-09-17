"""Main window and entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Dict

from PyQt5 import QtCore, QtGui, QtWidgets

from .session import Params, Session
from .structures import StructureLegend


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


class CubeMode(QtWidgets.QWidget):
    """The 3D cube plus its legend and camera controls."""

    def __init__(self, session: Session, animator, parent=None):
        super().__init__(parent)
        from .glview import CubeView

        from .flow import FlowLayout

        self.session = session
        self.view = CubeView(session, animator)
        self.view.setMinimumSize(200, 150)
        top = FlowLayout()
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
        self.color_combo.addItems(["raw 0/1", "changed since previous step", "avalanche difference", "dye"])
        self.color_combo.currentIndexChanged.connect(
            lambda i: session.set_color_mode(session.COLOR_MODES[i]))
        top.addWidget(self.color_combo)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("cells:"))
        self.style_combo = QtWidgets.QComboBox()
        for key, title in session.CELL_STYLES:
            self.style_combo.addItem(title, key)
        self.style_combo.setToolTip("How a 1 and a 0 are drawn")
        self.style_combo.currentIndexChanged.connect(lambda i: session.set_cell_style(self.style_combo.itemData(i)))
        top.addWidget(self.style_combo)
        self.vis_combo = QtWidgets.QComboBox()
        for key, title in session.VISIBILITY:
            self.vis_combo.addItem(title, key)
        self.vis_combo.setToolTip("Hide the 0 bits or the 1 bits")
        self.vis_combo.currentIndexChanged.connect(lambda i: session.set_visibility(self.vis_combo.itemData(i)))
        top.addWidget(self.vis_combo)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("lines:"))
        self.lines_combo = QtWidgets.QComboBox()
        self.lines_combo.addItems(["none", "focused (selected + tracked bits)", "all"])
        self.lines_combo.setCurrentIndex(1)
        self.lines_combo.setToolTip("How many feed lines / arrows to draw during step animations")
        self.lines_combo.currentIndexChanged.connect(lambda i: self.view.set_line_mode(self.view.LINE_MODES[i]))
        top.addWidget(self.lines_combo)
        self.words_cb = QtWidgets.QCheckBox("words as blocks")
        self.words_cb.setToolTip("Draw each word as one block labelled with its value (word size in the parameters)")
        self.words_cb.toggled.connect(session.set_show_words)
        session.wordsChanged.connect(lambda on: self.words_cb.setChecked(on))
        top.addWidget(self.words_cb)
        self.regions_cb = QtWidgets.QCheckBox("rate / capacity")
        self.regions_cb.setToolTip("Outline the rate lanes (green) and the capacity lanes (purple)")
        self.regions_cb.toggled.connect(session.set_show_regions)
        session.regionsChanged.connect(lambda on: self.regions_cb.setChecked(on))
        top.addWidget(self.regions_cb)
        self.display_btn = QtWidgets.QToolButton()
        self.display_btn.setText("display ▾")
        self.display_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self.display_btn)
        self.display_actions = {}
        for name, title in self.view.DISPLAY_TOGGLES:
            a = menu.addAction(title)
            a.setCheckable(True)
            a.setChecked(getattr(self.view, name))
            a.toggled.connect(lambda on, n=name: self.view.set_toggle(n, on))
            self.display_actions[name] = a
        self.display_btn.setMenu(menu)
        top.addWidget(self.display_btn)
        top.addStretch(1)
        # spacing ("extrude") sliders
        row2 = FlowLayout()
        row2.addWidget(QtWidgets.QLabel("spacing (extrude):"))
        self.spacing_sliders = []
        for axis, name in enumerate(("x", "y", "z")):
            row2.addWidget(QtWidgets.QLabel(name))
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(10, 50)
            sl.setValue(10)
            sl.setFixedWidth(110)
            sl.setToolTip(f"Distance between cells along {name} (1× … 5×)")
            sl.valueChanged.connect(lambda v, a=axis: self.view.set_spacing(a, v / 10.0))
            row2.addWidget(sl)
            self.spacing_sliders.append(sl)
        b = QtWidgets.QToolButton()
        b.setText("reset")
        b.setAutoRaise(True)
        b.clicked.connect(lambda: [sl.setValue(10) for sl in self.spacing_sliders])
        row2.addWidget(b)
        row2.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        topw = QtWidgets.QWidget()
        topw.setLayout(top)
        lay.addWidget(topw)
        row2w = QtWidgets.QWidget()
        row2w.setLayout(row2)
        lay.addWidget(row2w)
        lay.addWidget(self.view, 1)
        lay.addWidget(StructureLegend(session))
        session.colorModeChanged.connect(self._sync_color)

    def _sync_color(self, mode: str) -> None:
        i = self.session.COLOR_MODES.index(mode)
        if self.color_combo.currentIndex() != i:
            self.color_combo.setCurrentIndex(i)


class MainWindow(QtWidgets.QMainWindow):
    MODES = [
        ("cube", "3D cube"),
        ("slices", "Slice stack"),
        ("words", "Words"),
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
        self.setMinimumSize(640, 480)

        from .animator import StepAnimator
        from .transport import Transport

        self.animator = StepAnimator(session)
        self.transport = Transport(session)
        self.stack = QtWidgets.QStackedWidget()
        self.modes: Dict[str, QtWidgets.QWidget] = {}
        self.cube = CubeMode(session, self.animator)
        self._add_mode("cube", self.cube)
        self.transport.speedChanged.connect(self.animator.set_animation_ms)
        self.animator.set_animation_ms(self.transport.anim_ms)
        self.transport.scrubbed.connect(self.animator.freeze)
        self.animator.progress.connect(self.transport.set_progress)
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
        QtCore.QTimer.singleShot(0, self._check_gl)

    def _check_gl(self) -> None:
        """If the 3D view could not start, say so once and open a 2D view instead."""
        if getattr(self.cube.view, "gl_error", None) and self.current_mode == "cube":
            self.statusBar().showMessage("3D cube unavailable on this system (no OpenGL 3.3) - showing the slice "
                                         "stack; see the cube tab for what to try.", 15000)
            self.set_mode("slices")

    # ------------------------------------------------------------ modes

    def _add_mode(self, key: str, widget: QtWidgets.QWidget) -> None:
        self.modes[key] = widget
        self.stack.addWidget(widget)

    def _build_other_modes(self) -> None:
        from .modes import make_modes

        for key, widget in make_modes(self.session, self.animator):
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

        from .dyes import DyePanel

        self.dye_panel = DyePanel(self.session)
        d4 = QtWidgets.QDockWidget("Dyes", self)
        d4.setObjectName("dyes")
        d4.setWidget(self.dye_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d4)
        self.tabifyDockWidget(d3, d4)
        d2.raise_()
        self.resizeDocks([d1, d2], [380, 520], QtCore.Qt.Vertical)
        self.resizeDocks([d1, d2, d3, d4], [400, 400, 400, 400], QtCore.Qt.Horizontal)
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
        for i, (mode, title) in enumerate(zip(self.session.COLOR_MODES, ("raw 0/1", "changed", "avalanche", "dye"))):
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
        sc(["F"], lambda: s.track_focus(s.selected))
        sc(["Shift+F"], s.clear_tracked)
        sc(["D"], lambda: s.add_dye(s.selected))
        sc(["Shift+D"], s.clear_dyes)
        sc(["X"], lambda: s.pull_out(s.structure, s.selected))
        sc(["Shift+X"], lambda: s.push_back())
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
  F / Shift+F    track the selected bit (or the whole highlighted row/column/lane/slice/plane/sheet) / clear
  D / Shift+D    dye the selected bit or highlighted structure with a colour / clear dyes
  X / Shift+X    pull the highlighted structure out of the cube / push everything back
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
    p.add_argument("--style", default=None, help="cell style: cubes, equal, spheres, mono")
    p.add_argument("--visibility", default=None, help="both, ones or zeros")
    p.add_argument("--spacing", default=None, help="cell spacing x,y,z e.g. 1,1,2.5")
    p.add_argument("--dye", default=None, help="x,y,z[:structure] triples to dye, ';'-separated, or 'input'/'capacity'")
    p.add_argument("--pull", default=None, help="structure:x,y,z regions to pull out, ';'-separated")
    p.add_argument("--word-bits", type=int, default=None)
    p.add_argument("--words-per-cycle", type=int, default=None)
    p.add_argument("--no-load", action="store_true", help="hide the loading phase")
    p.add_argument("--extra", action="append", default=[], help="extra input absorbed after the message (repeatable)")
    p.add_argument("--squeeze", type=int, action="append", default=[], help="extra squeeze call of N bytes (repeatable)")
    p.add_argument("--output-bytes", type=int, default=None)
    p.add_argument("--regions", action="store_true", help="show the rate / capacity split")
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
    params = Params(message=msg, message_is_hex=args.hex, num_rounds=args.rounds, show_load=not args.no_load)
    params.extra_inputs = tuple(e.encode() for e in args.extra)
    if args.word_bits:
        params.word_bits = args.word_bits
    if args.words_per_cycle:
        params.words_per_cycle = args.words_per_cycle
    v = S.VARIANTS[args.variant]
    params.variant, params.rate_bytes, params.domain_byte = v.name, v.rate_bytes, v.domain_byte
    params.output_bytes = v.output_bytes or 32
    if args.output_bytes is not None:
        params.output_bytes = args.output_bytes

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
    if args.style:
        win.cube.style_combo.setCurrentIndex([k for k, _t in Session.CELL_STYLES].index(args.style))
    if args.spacing:
        for sl, v in zip(win.cube.spacing_sliders, args.spacing.split(",")):
            sl.setValue(int(float(v) * 10))
    for n in args.squeeze:
        session.add_squeeze(n)
    if args.regions:
        session.set_show_regions(True)
    if args.visibility:
        win.cube.vis_combo.setCurrentIndex([k for k, _t in Session.VISIBILITY].index(args.visibility))
    if args.dye:
        for item in args.dye.split(";"):
            if item in ("input", "capacity"):
                session.add_region_dye(item)
                continue
            cell, _, struct = item.partition(":")
            session.set_structure(struct or None)
            session.add_dye(tuple(int(v) for v in cell.split(",")))
        session.set_structure(None)
    if args.pull:
        for item in args.pull.split(";"):
            name, cell = item.split(":")
            session.pull_out(name, tuple(int(v) for v in cell.split(",")))
    if args.track:
        for trip in args.track.split(";"):
            if ":" in trip:  # structure:x,y,z
                name, cell = trip.split(":")
                session.track_structure(name, tuple(int(v) for v in cell.split(",")))
            else:
                session.track(tuple(int(v) for v in trip.split(",")))
    if args.anim_t is not None:
        win.animator.freeze(args.anim_t)

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
