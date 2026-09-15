"""The sponge construction on top of Keccak-p[1600].

Implements SHA3-224/256/384/512 (domain byte 0x06), SHAKE128/256 (0x1F) and
the original pre-standardisation Keccak (0x01), plus arbitrary rate /
capacity / round-count / step-toggle settings for experimentation.

Everything the sponge does is recorded in a :class:`SpongeRun` so the UI can
draw the absorb / squeeze timeline and jump into any permutation call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

from . import keccak as K


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    name: str
    rate_bytes: int
    domain_byte: int  # the suffix bits, as a byte with the first pad bit included
    output_bytes: Optional[int]  # None = extendable output (XOF)

    @property
    def capacity_bytes(self) -> int:
        return K.STATE_BYTES - self.rate_bytes

    @property
    def rate_bits(self) -> int:
        return self.rate_bytes * 8

    @property
    def capacity_bits(self) -> int:
        return self.capacity_bytes * 8

    @property
    def is_xof(self) -> bool:
        return self.output_bytes is None

    @property
    def security_bits(self) -> int:
        return self.capacity_bits // 2

    def with_(self, **kw) -> "Variant":
        d = dict(name=self.name, rate_bytes=self.rate_bytes,
                 domain_byte=self.domain_byte, output_bytes=self.output_bytes)
        d.update(kw)
        return Variant(**d)


DOMAIN_SHA3 = 0x06  # suffix 01, then pad bit
DOMAIN_SHAKE = 0x1F  # suffix 1111, then pad bit
DOMAIN_KECCAK = 0x01  # no suffix, only the pad bit (Keccak as submitted to SHA-3)

VARIANTS = {
    "SHA3-224": Variant("SHA3-224", 144, DOMAIN_SHA3, 28),
    "SHA3-256": Variant("SHA3-256", 136, DOMAIN_SHA3, 32),
    "SHA3-384": Variant("SHA3-384", 104, DOMAIN_SHA3, 48),
    "SHA3-512": Variant("SHA3-512", 72, DOMAIN_SHA3, 64),
    "SHAKE128": Variant("SHAKE128", 168, DOMAIN_SHAKE, None),
    "SHAKE256": Variant("SHAKE256", 136, DOMAIN_SHAKE, None),
    "Keccak-224": Variant("Keccak-224", 144, DOMAIN_KECCAK, 28),
    "Keccak-256": Variant("Keccak-256", 136, DOMAIN_KECCAK, 32),
    "Keccak-384": Variant("Keccak-384", 104, DOMAIN_KECCAK, 48),
    "Keccak-512": Variant("Keccak-512", 72, DOMAIN_KECCAK, 64),
}

DOMAIN_NAMES = {DOMAIN_SHA3: "SHA-3 (0x06)", DOMAIN_SHAKE: "SHAKE (0x1F)", DOMAIN_KECCAK: "Keccak (0x01)"}


# --------------------------------------------------------------------------
# Padding
# --------------------------------------------------------------------------


def pad(message: bytes, rate_bytes: int, domain_byte: int) -> bytes:
    """Multi-rate padding pad10*1 with the domain-separation suffix.

    The suffix bits and the first '1' pad bit are merged into ``domain_byte``
    (0x06 for SHA-3, 0x1F for SHAKE, 0x01 for plain Keccak).  The final '1'
    pad bit is the top bit (0x80) of the last byte of the block; when the
    message fills the block up to its last byte, both land in the same byte.
    """
    if not 1 <= rate_bytes <= K.STATE_BYTES:
        raise ValueError("rate must be 1..200 bytes")
    if not 0 < domain_byte < 0x80:
        raise ValueError("domain byte must have its pad bit below 0x80")
    q = rate_bytes - (len(message) % rate_bytes)
    if q == 1:
        return message + bytes([domain_byte | 0x80])
    return message + bytes([domain_byte]) + bytes(q - 2) + bytes([0x80])


def padding_bytes(message_len: int, rate_bytes: int, domain_byte: int) -> bytes:
    """Only the appended padding, for display."""
    return pad(bytes(message_len), rate_bytes, domain_byte)[message_len:]


# --------------------------------------------------------------------------
# Sponge with full recording
# --------------------------------------------------------------------------


@dataclass
class PermCall:
    """One permutation call in the run."""

    index: int  # 0-based position in the run
    phase: str  # "absorb" or "squeeze"
    block_index: int  # which absorb block / squeeze block triggered it
    state_in: np.ndarray
    state_out: np.ndarray
    trace: Optional[K.Trace] = None


@dataclass
class AbsorbBlock:
    index: int
    data: bytes  # rate_bytes long, padded
    is_last: bool
    state_before: np.ndarray
    state_after_xor: np.ndarray
    perm: PermCall


@dataclass
class SqueezeBlock:
    index: int
    data: bytes  # up to rate_bytes long (truncated at the end)
    state: np.ndarray  # state the bytes were read from
    perm: Optional[PermCall]  # permutation run *after* reading (None for the last)


@dataclass
class SpongeRun:
    variant: Variant
    message: bytes
    padded: bytes
    num_rounds: int
    enabled_steps: Tuple[str, ...]
    output_bytes: int
    absorb_blocks: List[AbsorbBlock] = field(default_factory=list)
    squeeze_blocks: List[SqueezeBlock] = field(default_factory=list)
    perm_calls: List[PermCall] = field(default_factory=list)
    output: bytes = b""

    @property
    def num_perm_calls(self) -> int:
        return len(self.perm_calls)

    @property
    def padding(self) -> bytes:
        return self.padded[len(self.message):]

    def timeline(self) -> List[Tuple[str, int]]:
        """Ordered (event, index) list: ("absorb", i) / ("perm", k) / ("squeeze", j)."""
        ev: List[Tuple[str, int]] = []
        for b in self.absorb_blocks:
            ev.append(("absorb", b.index))
            ev.append(("perm", b.perm.index))
        for s in self.squeeze_blocks:
            ev.append(("squeeze", s.index))
            if s.perm is not None:
                ev.append(("perm", s.perm.index))
        return ev


# --------------------------------------------------------------------------
# Loading phase: the block enters the state word by word over a bus
# --------------------------------------------------------------------------


@dataclass
class FullTrace(K.Trace):
    """A permutation trace with the loading of its input block prepended.

    Snapshot 0 shows the incoming block alone ("seed"), snapshot 1 the state
    register before the absorb ("initial value"), then one snapshot per bus
    cycle as words are XORed in; the permutation snapshots follow with their
    indices shifted by ``n_load``."""

    n_load: int = 0
    core: Optional[K.Trace] = None

    def core_index(self, i: int) -> int:
        return max(0, i - self.n_load)

    def index_of(self, round_: int, step: str) -> int:
        return self.n_load + 1 + 5 * round_ + K.STEP_NAMES.index(step)

    def round_end_indices(self) -> List[int]:
        return [self.index_of(r, "iota") for r in range(self.num_rounds)]


def word_cells(word: int, word_bits: int) -> List[Tuple[int, int, int]]:
    """Cells (x, y, z) of word number ``word`` in the standard bit-string order."""
    return [K.bit_coords(i) for i in range(word * word_bits, (word + 1) * word_bits)]


def load_snapshots(state_before: np.ndarray, block: bytes, word_bits: int = 64,
                   words_per_cycle: int = 1) -> List[K.Snapshot]:
    """Snapshots of the block being XORed into the state over a bus that
    delivers ``words_per_cycle`` words of ``word_bits`` bits per cycle."""
    if 64 % word_bits and word_bits % 64:
        raise ValueError("word size must divide 64 or be a multiple of 64")
    block_bits = len(block) * 8
    n_words = -(-block_bits // word_bits)
    per_cycle = max(1, words_per_cycle)
    n_cycles = -(-n_words // per_cycle)
    block_state = K.state_from_bytes(block + bytes(K.STATE_BYTES - len(block)))
    snaps = [
        K.Snapshot(0, -1, "load", block_state.copy(),
                   info={"phase": "seed", "cycle": 0, "n_cycles": n_cycles, "word_bits": word_bits,
                         "words": [], "cells": [], "n_words": n_words}),
        K.Snapshot(1, -1, "load", np.array(state_before, dtype=np.uint64, copy=True),
                   info={"phase": "iv", "cycle": 0, "n_cycles": n_cycles, "word_bits": word_bits,
                         "words": [], "cells": [], "n_words": n_words}),
    ]
    buf = bytearray(K.state_to_bytes(state_before))
    block_bits_arr = K.state_to_flat_bits(block_state)
    for c in range(n_cycles):
        words = list(range(c * per_cycle, min(n_words, (c + 1) * per_cycle)))
        cells = []
        for w in words:
            for i in range(w * word_bits, min(block_bits, (w + 1) * word_bits)):
                if block_bits_arr[i]:
                    buf[i // 8] ^= 1 << (i % 8)
                cells.append(i)
        snaps.append(K.Snapshot(2 + c, -1, "load", K.state_from_bytes(bytes(buf)),
                                info={"phase": "bus", "cycle": c + 1, "n_cycles": n_cycles, "word_bits": word_bits,
                                      "words": words, "cells": cells, "n_words": n_words}))
    return snaps


def full_trace(call: "PermCall", block: Optional["AbsorbBlock"], word_bits: int = 64,
               words_per_cycle: int = 1) -> FullTrace:
    """The permutation trace of ``call`` with the loading phase in front (absorb
    calls only; squeeze calls get an empty loading phase)."""
    import dataclasses

    core = call.trace
    load = load_snapshots(block.state_before, block.data, word_bits, words_per_cycle) if block else []
    n = len(load)
    snaps = load + [dataclasses.replace(s, index=s.index + n) for s in core.snapshots]
    return FullTrace(snapshots=snaps, num_rounds=core.num_rounds, enabled_steps=core.enabled_steps,
                     round_offset=core.round_offset, n_load=n, core=core)


def _xor_block_into(state: np.ndarray, block: bytes) -> np.ndarray:
    buf = bytearray(K.state_to_bytes(state))
    for i, b in enumerate(block):
        buf[i] ^= b
    return K.state_from_bytes(bytes(buf))


def sponge(
    message: bytes,
    variant: Variant,
    output_bytes: Optional[int] = None,
    num_rounds: int = K.NUM_ROUNDS,
    enabled_steps: Iterable[str] = K.STEP_NAMES,
    trace: bool = True,
    round_offset: Optional[int] = None,
) -> SpongeRun:
    """Run the full sponge and return a :class:`SpongeRun` record.

    ``output_bytes`` defaults to the variant's fixed digest length, or 32 for
    XOFs.  ``trace=False`` skips per-step snapshots (fast path for tests)."""
    if output_bytes is None:
        output_bytes = variant.output_bytes if variant.output_bytes is not None else 32
    if output_bytes < 0:
        raise ValueError("output length must be >= 0")
    enabled = tuple(s for s in K.STEP_NAMES if s in set(enabled_steps))
    r = variant.rate_bytes
    padded = pad(message, r, variant.domain_byte)
    run = SpongeRun(variant, message, padded, num_rounds, enabled, output_bytes)

    def do_perm(state, phase, block_index):
        if trace:
            new, tr = K.permute(state, num_rounds, trace=True, enabled_steps=enabled,
                                round_offset=round_offset)
        else:
            new, tr = K.permute(state, num_rounds, enabled_steps=enabled,
                                round_offset=round_offset), None
        call = PermCall(len(run.perm_calls), phase, block_index, state.copy(), new.copy(), tr)
        run.perm_calls.append(call)
        return new, call

    state = K.new_state()
    nblocks = len(padded) // r
    for i in range(nblocks):
        block = padded[i * r:(i + 1) * r]
        before = state.copy()
        state = _xor_block_into(state, block)
        after_xor = state.copy()
        state, call = do_perm(state, "absorb", i)
        run.absorb_blocks.append(AbsorbBlock(i, block, i == nblocks - 1, before, after_xor, call))

    out = bytearray()
    j = 0
    while len(out) < output_bytes:
        chunk = K.state_to_bytes(state)[:r]
        take = min(r, output_bytes - len(out))
        out += chunk[:take]
        need_more = len(out) < output_bytes
        call = None
        if need_more:
            state, call = do_perm(state, "squeeze", j)
        run.squeeze_blocks.append(SqueezeBlock(j, chunk[:take], state if call is None else call.state_in, call))
        j += 1
    if output_bytes == 0:
        # still record that the state was "read" zero times; nothing to do
        pass
    run.output = bytes(out)
    return run


def digest(message: bytes, variant: Variant, output_bytes: Optional[int] = None,
           num_rounds: int = K.NUM_ROUNDS) -> bytes:
    """Hash without tracing."""
    return sponge(message, variant, output_bytes, num_rounds, trace=False).output


def sha3_224(m: bytes) -> bytes:
    return digest(m, VARIANTS["SHA3-224"])


def sha3_256(m: bytes) -> bytes:
    return digest(m, VARIANTS["SHA3-256"])


def sha3_384(m: bytes) -> bytes:
    return digest(m, VARIANTS["SHA3-384"])


def sha3_512(m: bytes) -> bytes:
    return digest(m, VARIANTS["SHA3-512"])


def shake128(m: bytes, n: int) -> bytes:
    return digest(m, VARIANTS["SHAKE128"], n)


def shake256(m: bytes, n: int) -> bytes:
    return digest(m, VARIANTS["SHAKE256"], n)


def keccak256(m: bytes) -> bytes:
    return digest(m, VARIANTS["Keccak-256"])


__all__ = [
    "Variant", "VARIANTS", "DOMAIN_SHA3", "DOMAIN_SHAKE", "DOMAIN_KECCAK", "DOMAIN_NAMES",
    "pad", "padding_bytes", "PermCall", "AbsorbBlock", "SqueezeBlock", "SpongeRun",
    "sponge", "digest", "sha3_224", "sha3_256", "sha3_384", "sha3_512",
    "shake128", "shake256", "keccak256", "FullTrace", "load_snapshots", "full_trace", "word_cells",
]
