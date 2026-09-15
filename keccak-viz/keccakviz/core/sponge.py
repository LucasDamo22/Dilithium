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
    "shake128", "shake256", "keccak256",
]
