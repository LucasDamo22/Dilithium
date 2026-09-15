"""Live parameter panel: variant, domain byte, rate, rounds, output length,
message, step toggles, analysis bits."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ..core import keccak as K
from ..core import sponge as S
from .session import Session


class HexSpin(QtWidgets.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDisplayIntegerBase(16)
        self.setPrefix("0x")

    def textFromValue(self, v: int) -> str:
        return f"{v:02x}"


class ParamsPanel(QtWidgets.QScrollArea):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        self.setWidget(inner)
        form = QtWidgets.QFormLayout(inner)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setContentsMargins(8, 8, 8, 8)
        self._updating = False

        self.variant = QtWidgets.QComboBox()
        self.variant.addItems(list(S.VARIANTS))
        self.variant.setToolTip("Standard variant: sets rate, capacity, domain byte and output length")
        self.variant.currentTextChanged.connect(self._on_variant)
        form.addRow("variant", self.variant)

        self.domain = HexSpin()
        self.domain.setRange(1, 0x7F)
        self.domain.setToolTip("Domain-separation suffix merged with the first pad bit: 0x06 SHA-3, 0x1F SHAKE, 0x01 Keccak")
        self.domain.valueChanged.connect(lambda v: self._set(domain_byte=v))
        form.addRow("domain byte", self.domain)

        self.rate = QtWidgets.QSpinBox()
        self.rate.setRange(8, 200)
        self.rate.setSingleStep(8)
        self.rate.setSuffix(" bytes")
        self.rate.setToolTip("Rate r in bytes (non-standard values allowed for experiments); capacity = 200 - r")
        self.rate.valueChanged.connect(self._on_rate)
        form.addRow("rate", self.rate)
        self.capacity = QtWidgets.QLabel()
        form.addRow("capacity", self.capacity)

        self.rounds = QtWidgets.QSpinBox()
        self.rounds.setRange(1, 24)
        self.rounds.setToolTip("Number of rounds (reduced-round Keccak-p)")
        self.rounds.valueChanged.connect(lambda v: self._set(num_rounds=v))
        form.addRow("rounds", self.rounds)
        self.rc_std = QtWidgets.QCheckBox("use the last n round constants (FIPS 202 Keccak-p)")
        self.rc_std.setToolTip("Unchecked: rounds use RC[0..n-1] instead of RC[24-n..23]")
        self.rc_std.toggled.connect(lambda on: self._set(round_offset_standard=on))
        form.addRow("", self.rc_std)

        self.out_len = QtWidgets.QSpinBox()
        self.out_len.setRange(0, 4096)
        self.out_len.setSuffix(" bytes")
        self.out_len.setToolTip("Output length; more than one rate block of output means extra squeeze permutations")
        self.out_len.valueChanged.connect(lambda v: self._set(output_bytes=v))
        form.addRow("output length", self.out_len)
        self.sq_step = QtWidgets.QSpinBox()
        self.sq_step.setRange(1, 4096)
        self.sq_step.setSuffix(" bytes")
        self.sq_step.setToolTip("How much each “+ squeeze” asks for")
        self.sq_step.valueChanged.connect(lambda v: self._set(squeeze_step=v))
        self.sq_label = QtWidgets.QLabel()
        self.sq_reset = QtWidgets.QPushButton("reset")
        self.sq_reset.setToolTip("Back to a single squeeze call of the output length")
        self.sq_reset.clicked.connect(session.reset_squeezes)
        sqrow = QtWidgets.QHBoxLayout()
        sqrow.addWidget(self.sq_step)
        sqrow.addWidget(self.sq_label, 1)
        sqrow.addWidget(self.sq_reset)
        form.addRow("bytes per squeeze", sqrow)

        self.message = QtWidgets.QLineEdit()
        self.message.setToolTip("Input message (text, or hex when the box is ticked)")
        self.message.editingFinished.connect(self._on_message)
        self.hex_cb = QtWidgets.QCheckBox("hex")
        self.hex_cb.toggled.connect(self._on_hex)
        mrow = QtWidgets.QHBoxLayout()
        mrow.addWidget(self.message, 1)
        mrow.addWidget(self.hex_cb)
        form.addRow("message", mrow)

        self.word_bits = QtWidgets.QComboBox()
        for w in session.WORD_SIZES:
            self.word_bits.addItem(f"{w} bits", w)
        self.word_bits.setToolTip("Word size for the Words view, word bands and the loading bus")
        self.word_bits.currentIndexChanged.connect(lambda i: self._set(word_bits=self.word_bits.itemData(i)))
        form.addRow("word size", self.word_bits)
        self.wpc = QtWidgets.QSpinBox()
        self.wpc.setRange(1, 1600)
        self.wpc.setToolTip("Words delivered per bus cycle while loading a block (bus width = words × word size)")
        self.wpc.valueChanged.connect(lambda v: self._set(words_per_cycle=v))
        form.addRow("words per load cycle", self.wpc)
        self.bus_label = QtWidgets.QLabel()
        form.addRow("bus width", self.bus_label)
        self.show_load = QtWidgets.QCheckBox("show the loading phase before each absorb")
        self.show_load.toggled.connect(lambda on: self._set(show_load=on))
        form.addRow("", self.show_load)

        self.extra_label = QtWidgets.QLabel()
        self.extra_clear = QtWidgets.QPushButton("remove")
        self.extra_clear.setToolTip("Remove all input added after the message")
        self.extra_clear.clicked.connect(session.clear_inputs)
        erow = QtWidgets.QHBoxLayout()
        erow.addWidget(self.extra_label, 1)
        erow.addWidget(self.extra_clear)
        form.addRow("extra inputs", erow)
        self.show_squeeze = QtWidgets.QCheckBox("show the squeeze read-out after the permutation")
        self.show_squeeze.toggled.connect(lambda on: self._set(show_squeeze=on))
        form.addRow("", self.show_squeeze)

        steps_box = QtWidgets.QGroupBox("step mappings - untick one to disable it, then watch the avalanche plot")
        sl = QtWidgets.QGridLayout(steps_box)
        self.step_cbs = {}
        for i, name in enumerate(K.STEP_NAMES):
            cb = QtWidgets.QCheckBox(f"{K.STEP_SYMBOLS[name]} {name}")
            cb.setChecked(True)
            cb.toggled.connect(self._on_steps)
            sl.addWidget(cb, i // 3, i % 3)
            self.step_cbs[name] = cb
        form.addRow(steps_box)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        f.setPointSize(8)
        self.summary.setFont(f)
        form.addRow(self.summary)

        session.runChanged.connect(self.sync)
        self.sync()

    # ------------------------------------------------------------

    def _set(self, **kw) -> None:
        if not self._updating:
            self.session.set_params(**kw)

    def _on_variant(self, name: str) -> None:
        if not self._updating and name:
            self.session.apply_variant(name)

    def _on_rate(self, v: int) -> None:
        if not self._updating:
            self.capacity.setText(f"{200 - v} bytes ({(200 - v) * 8} bits) → {(200 - v) * 4}-bit security")
            self.session.set_params(rate_bytes=v)

    def _on_message(self) -> None:
        if self._updating:
            return
        txt = self.message.text()
        if self.hex_cb.isChecked():
            try:
                msg = bytes.fromhex(txt.replace(" ", ""))
            except ValueError:
                self.message.setStyleSheet("background: #5a2a2a;")
                return
        else:
            msg = txt.encode("utf-8")
        self.message.setStyleSheet("")
        self.session.set_params(message=msg)

    def _on_hex(self, on: bool) -> None:
        if self._updating:
            return
        p = self.session.params
        self.message.setText(p.message.hex() if on else p.message.decode("utf-8", "replace"))
        self.session.set_params(message_is_hex=on)

    def _on_steps(self, _on: bool) -> None:
        if self._updating:
            return
        enabled = tuple(n for n in K.STEP_NAMES if self.step_cbs[n].isChecked())
        self.session.set_params(enabled_steps=enabled)

    def sync(self) -> None:
        self._updating = True
        try:
            p = self.session.params
            self.variant.setCurrentText(p.variant)
            self.domain.setValue(p.domain_byte)
            self.rate.setValue(p.rate_bytes)
            self.capacity.setText(f"{200 - p.rate_bytes} bytes ({(200 - p.rate_bytes) * 8} bits) → "
                                  f"{(200 - p.rate_bytes) * 4}-bit security")
            self.rounds.setValue(p.num_rounds)
            self.rc_std.setChecked(p.round_offset_standard)
            self.out_len.setValue(p.output_bytes)
            self.out_len.setSuffix(f" bytes = {p.output_bytes * 8} bits")
            self.sq_step.setValue(p.squeeze_step)
            n_req = len(self.session.squeeze_requests())
            self.sq_label.setText(f"{n_req} squeeze call(s)")
            self.sq_reset.setEnabled(bool(p.squeeze_sizes))
            self.hex_cb.setChecked(p.message_is_hex)
            if not self.message.hasFocus():
                self.message.setText(p.message.hex() if p.message_is_hex else p.message.decode("utf-8", "replace"))
            for n, cb in self.step_cbs.items():
                cb.setChecked(n in p.enabled_steps)
            self.word_bits.setCurrentIndex(list(self.session.WORD_SIZES).index(p.word_bits))
            self.wpc.setValue(p.words_per_cycle)
            bus = p.word_bits * p.words_per_cycle
            cycles = -(-p.rate_bytes * 8 // bus)
            self.bus_label.setText(f"{bus} bits/cycle → {cycles} cycle(s) per {p.rate_bytes * 8}-bit block")
            self.show_load.setChecked(p.show_load)
            self.show_squeeze.setChecked(p.show_squeeze)
            n_extra = len(p.extra_inputs)
            self.extra_label.setText("none (use “+ input” at the bottom)" if not n_extra else
                                     f"{n_extra} added: " + ", ".join(f"{len(e)} B" for e in p.extra_inputs)
                                     + " - output is no longer SHA3(message)")
            self.extra_clear.setEnabled(bool(n_extra))
            run = self.session.run
            std = S.VARIANTS[p.variant]
            is_std = (p.rate_bytes == std.rate_bytes and p.domain_byte == std.domain_byte and p.num_rounds == 24
                      and len(p.enabled_steps) == 5 and (std.output_bytes in (None, p.output_bytes))
                      and not p.extra_inputs)
            self.summary.setText(
                f"{'standard ' + p.variant if is_std else 'NON-STANDARD parameters'}\n"
                f"{len(run.message)} B message → {len(run.padded) // p.rate_bytes} block(s), "
                f"{run.num_perm_calls} permutation call(s)\n"
                f"output: {run.output.hex()[:64]}{'…' if len(run.output) > 32 else ''}")
        finally:
            self._updating = False
