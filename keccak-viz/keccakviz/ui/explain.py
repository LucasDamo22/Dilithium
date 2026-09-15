"""Side panel with a Plain / Student / Expert slider whose text follows the
current step mapping and the current view."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..core import keccak as K
from .explain_text import GLOSSARY, LEVEL_NAMES, MODE_TEXT, STEP_TEXT
from .session import Session

CSS = """
<style>
body { color: #e1e1e6; font-size: 10pt; }
h3 { color: #ffd166; margin: 8px 0 4px 0; }
h4 { color: #9ad0ff; margin: 10px 0 2px 0; }
pre { background: #1c1e24; color: #cfe; padding: 6px; font-size: 9pt; }
code { color: #cfe; }
.facts { background: #2c2f38; padding: 6px; border-left: 3px solid #ffd166; }
</style>
"""


class ExplainPanel(QtWidgets.QWidget):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.mode = "cube"
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("detail:"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 2)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setValue(session.detail_level)
        self.slider.valueChanged.connect(session.set_detail_level)
        row.addWidget(self.slider, 1)
        self.level_label = QtWidgets.QLabel(LEVEL_NAMES[session.detail_level])
        self.level_label.setMinimumWidth(60)
        row.addWidget(self.level_label)
        lay.addLayout(row)
        self.browser = QtWidgets.QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        lay.addWidget(self.browser, 1)
        session.positionChanged.connect(lambda *_: self.refresh())
        session.runChanged.connect(self.refresh)
        session.detailLevelChanged.connect(self._on_level)
        self.refresh()

    def _on_level(self, level: int) -> None:
        self.level_label.setText(LEVEL_NAMES[level])
        if self.slider.value() != level:
            self.slider.setValue(level)
        self.refresh()

    def set_mode(self, key: str) -> None:
        self.mode = key
        self.refresh()

    def _facts(self) -> str:
        s = self.session
        snap = s.snapshot
        p = s.params
        lvl = s.detail_level
        if snap.step == "initial":
            hw = K.hamming_weight(snap.state)
            return (f"<div class='facts'>Permutation call {s.perm_index + 1} of {s.run.num_perm_calls} "
                    f"({s.perm.phase}). Initial state: {hw} of 1600 bits set.</div>")
        if snap.step == "squeeze":
            info = snap.info or {}
            hx = info.get("data_hex", "")
            a0 = info.get("start_bit", 0)
            words = s.format_words(bytes.fromhex(hx), a0, 8)
            return (f"<div class='facts'>Squeeze call {info.get('request', 0) + 1}: {info.get('nbits')} output bits, "
                    f"rate bits {a0}…{a0 + info.get('nbits', 0) - 1}.<br>bytes <code>{hx[:64]}"
                    f"{'…' if len(hx) > 64 else ''}</code><br>{s.params.word_bits}-bit words <code>{words}</code>"
                    f"<br>Press <b>+ squeeze</b> for more: it continues in the rate; the permutation runs again "
                    f"only when the rate is used up.</div>")
        if snap.step == "load":
            info = snap.info or {}
            phase = info.get("phase")
            if phase == "seed":
                return "<div class='facts'>The padded block by itself, on the bus. Nothing has been loaded.</div>"
            if phase == "iv":
                hw = K.hamming_weight(snap.state)
                return (f"<div class='facts'>The state register before the absorb: {hw} of 1600 bits set "
                        f"({'all zero: first block' if hw == 0 else 'output of the previous permutation'}).</div>")
            return (f"<div class='facts'>Bus cycle {info.get('cycle')} of {info.get('n_cycles')}: "
                    f"{len(info.get('words', []))} word(s) of {info.get('word_bits')} bits XORed into the rate "
                    f"({len(info.get('cells', []))} bit positions).</div>")
        prev = s.prev_snapshot
        flipped = K.hamming_weight(snap.state ^ prev.state)
        lanes = int((snap.state != prev.state).sum())
        extra = ""
        if snap.skipped:
            extra = " <b>This step is disabled in the parameters, so nothing changed.</b>"
        elif snap.step == "iota":
            extra = f" RC[{snap.round_index_abs}] = 0x{snap.round_constant:016x}."
        elif snap.step == "theta" and lvl > 0:
            c_on = K.hamming_weight(snap.theta_c) if snap.theta_c is not None else 0
            d_on = K.hamming_weight(snap.theta_d) if snap.theta_d is not None else 0
            extra = f" {c_on} of 320 column parities are 1; {d_on} columns (of 320) flip."
        elif snap.step in ("rho", "pi") and lvl > 0:
            extra = " (bits only move; the number of ones is unchanged)"
        return (f"<div class='facts'>Round {snap.round + 1} of {p.num_rounds}, step "
                f"{K.STEP_SYMBOLS[snap.step]} <b>{snap.step}</b>: {flipped} bits in {lanes} lanes differ from the "
                f"previous snapshot.{extra}</div>")

    def refresh(self) -> None:
        s = self.session
        lvl = s.detail_level
        step = s.snapshot.step
        html = [CSS, self._facts()]
        html.append(f"<h3>{K.STEP_SYMBOLS.get(step, '')} {step}</h3>")
        html.append(STEP_TEXT[step][lvl])
        if step != "initial" and lvl >= 1:
            nxt = None
            if s.snap_index + 1 < s.num_snapshots:
                nxt = s.trace[s.snap_index + 1].step
            if nxt:
                html.append(f"<p style='color:#9aa'>Next: {K.STEP_SYMBOLS[nxt]} {nxt} - "
                            f"<code>{K.step_description(nxt)}</code></p>")
        html.append(f"<h4>About this view</h4>{MODE_TEXT[self.mode][lvl]}")
        if lvl >= 1 and self.mode in ("cube", "slices"):
            html.append(GLOSSARY)
        if lvl == 2 and step == "initial":
            html.append(ROUND_COST_TABLE)
        sb = self.browser.verticalScrollBar().value()
        self.browser.setHtml("".join(html))
        self.browser.verticalScrollBar().setValue(sb)


ROUND_COST_TABLE = """
<h4>Per-round cost sheet (two-input gates, full-width single-round datapath)</h4>
<table cellpadding="3" style="font-size:9pt; color:#ddd">
<tr><th>step</th><th>XOR2</th><th>AND2</th><th>depth</th><th>note</th></tr>
<tr><td>θ</td><td>≈3200</td><td>0</td><td>~5</td><td>1280 for C, 320 for D, 1600 for the update; deepest step</td></tr>
<tr><td>ρ</td><td>0</td><td>0</td><td>0</td><td>wiring</td></tr>
<tr><td>π</td><td>0</td><td>0</td><td>0</td><td>wiring</td></tr>
<tr><td>χ</td><td>1600</td><td>1600</td><td>2</td><td>only nonlinear gates; masking cost lives here</td></tr>
<tr><td>ι</td><td>≤7</td><td>0</td><td>1</td><td>merged into χ's XOR in practice</td></tr>
<tr><td><b>round</b></td><td>≈4800</td><td>1600</td><td>~8</td><td>+1600 state flip-flops; 24 cycles per permutation at one round per cycle</td></tr>
</table>
<p style="color:#9aa">Typical results: ~10–15 kGE for a compact 1-round/cycle core, 2 rounds/cycle roughly doubles logic
for ~1.8× throughput. The 1600-bit rate-width XOR for absorb adds up to 1344 XOR2 outside the round.</p>
"""
