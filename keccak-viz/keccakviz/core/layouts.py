"""Alternative bit layouts: bit-interleaving (32-bit targets) and batched
lanes (SIMD / vector registers).

Bit interleaving
----------------
A 64-bit lane L is stored as two 32-bit words:

    E = bits 0, 2, 4, ... 62 of L   (even word)
    O = bits 1, 3, 5, ... 63 of L   (odd word)

A 64-bit rotation ROL64(L, r) becomes two 32-bit rotations:

    r even:  E' = ROL32(E, r/2)        O' = ROL32(O, r/2)
    r odd:   E' = ROL32(O, (r+1)/2)    O' = ROL32(E, (r-1)/2)

so rho never needs a 64-bit shifter; theta's ROT(C, 1) becomes a swap plus a
single-bit rotate of one word.  Everything else (XOR, AND, NOT) is bitwise
and therefore layout-agnostic.

Batched lanes
-------------
N independent sponge instances.  Instead of one register per lane of one
state, register (x, y) holds lane (x, y) of *every* instance:

    R[x, y] = [ A_0[x, y], A_1[x, y], ..., A_{N-1}[x, y] ]

Theta's column parity C[x] = XOR_y A[x, y] is then five register-wide XORs
with no reduction across vector elements, rho is a per-element 64-bit rotate,
pi is register renaming and chi is register-wide logic: the whole
permutation is element-wise, which is why Keccak vectorises so cleanly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from . import keccak as K

M32 = 0xFFFFFFFF

# --------------------------------------------------------------------------
# bit interleaving
# --------------------------------------------------------------------------


def interleave(lane: int) -> Tuple[int, int]:
    """64-bit lane -> (even_word, odd_word), each 32 bits."""
    e = o = 0
    for i in range(32):
        e |= ((lane >> (2 * i)) & 1) << i
        o |= ((lane >> (2 * i + 1)) & 1) << i
    return e, o


def deinterleave(even: int, odd: int) -> int:
    lane = 0
    for i in range(32):
        lane |= ((even >> i) & 1) << (2 * i)
        lane |= ((odd >> i) & 1) << (2 * i + 1)
    return lane


def rol32(w: int, n: int) -> int:
    n %= 32
    return ((w << n) | (w >> (32 - n))) & M32 if n else w & M32


def rol64_interleaved(even: int, odd: int, r: int) -> Tuple[int, int]:
    """ROL64 expressed on the interleaved pair."""
    r %= 64
    if r % 2 == 0:
        return rol32(even, r // 2), rol32(odd, r // 2)
    return rol32(odd, (r + 1) // 2), rol32(even, (r - 1) // 2)


def interleave_rotation_plan(r: int) -> dict:
    """Describe how ROL64 by ``r`` decomposes, for display."""
    r %= 64
    if r % 2 == 0:
        return {"r": r, "swap": False, "even_rot": r // 2, "odd_rot": r // 2,
                "even_from": "E", "odd_from": "O"}
    return {"r": r, "swap": True, "even_rot": (r + 1) // 2, "odd_rot": (r - 1) // 2,
            "even_from": "O", "odd_from": "E"}


def state_interleaved(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(5,5) uint64 -> two (5,5) uint32 arrays (even words, odd words)."""
    bits = K.lanes_to_bits(state)  # (5,5,64)
    w = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    even = (bits[..., 0::2].astype(np.uint32) * w).sum(axis=-1, dtype=np.uint64).astype(np.uint32)
    odd = (bits[..., 1::2].astype(np.uint32) * w).sum(axis=-1, dtype=np.uint64).astype(np.uint32)
    return even, odd


# --------------------------------------------------------------------------
# batched lanes
# --------------------------------------------------------------------------


def batch_states(states) -> np.ndarray:
    """List of (5,5) states -> (N,5,5) array; the vector register for lane
    (x,y) is ``batch[:, x, y]``."""
    return np.stack([np.asarray(s, dtype=np.uint64) for s in states])


def register(batch: np.ndarray, x: int, y: int) -> np.ndarray:
    """The vector register holding lane (x,y) of every instance: shape (N,)."""
    return batch[:, x, y]


def theta_parity_registers(batch: np.ndarray) -> np.ndarray:
    """C[x] as five element-wise XORs of registers: shape (N, 5)."""
    return np.bitwise_xor.reduce(batch, axis=2)  # over y


def theta_parity_is_elementwise(batch: np.ndarray) -> bool:
    """Sanity check: the batched parity equals each instance's own parity."""
    c = theta_parity_registers(batch)
    for i in range(batch.shape[0]):
        _, ci, _ = K.theta_parts(batch[i])
        if not np.array_equal(c[i], ci):
            return False
    return True


__all__ = [
    "interleave", "deinterleave", "rol32", "rol64_interleaved", "interleave_rotation_plan",
    "state_interleaved", "batch_states", "register", "theta_parity_registers",
    "theta_parity_is_elementwise",
]
