"""Diffusion analyses built on the trace: avalanche and per-bit diffusion.

All functions are GUI-free and operate on :class:`keccak.Trace` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from . import keccak as K


def flip_bit(state: np.ndarray, x: int, y: int, z: int) -> np.ndarray:
    out = np.array(state, dtype=np.uint64, copy=True)
    out[x, y] ^= np.uint64(1) << np.uint64(z)
    return out


@dataclass
class Avalanche:
    """Two permutation runs that differ in the input, and their XOR."""

    base: K.Trace
    flipped: K.Trace
    flipped_bits: List[Tuple[int, int, int]]

    @property
    def num_snapshots(self) -> int:
        return len(self.base)

    def diff(self, index: int) -> np.ndarray:
        """(5, 5) uint64 XOR difference after snapshot ``index``."""
        return self.base[index].state ^ self.flipped[index].state

    def diff_bits(self, index: int) -> np.ndarray:
        return K.lanes_to_bits(self.diff(index))

    def hamming(self, index: int) -> int:
        return K.hamming_weight(self.diff(index))

    def hamming_per_snapshot(self) -> np.ndarray:
        return np.array([self.hamming(i) for i in range(len(self.base))])

    def hamming_per_round(self) -> np.ndarray:
        """Hamming distance after each full round, with the input as entry 0."""
        idx = [0] + self.base.round_end_indices()
        return np.array([self.hamming(i) for i in idx])


def avalanche(
    state: np.ndarray,
    flip: Iterable[Tuple[int, int, int]] = ((0, 0, 0),),
    num_rounds: int = K.NUM_ROUNDS,
    enabled_steps: Iterable[str] = K.STEP_NAMES,
    round_offset: Optional[int] = None,
    base_trace: Optional[K.Trace] = None,
) -> Avalanche:
    """Run the permutation on ``state`` and on ``state`` with the given bits
    flipped, returning both traces."""
    flips = list(flip)
    enabled = tuple(enabled_steps)
    if base_trace is None:
        _, base_trace = K.permute(state, num_rounds, trace=True, enabled_steps=enabled,
                                  round_offset=round_offset)
    other = np.array(state, dtype=np.uint64, copy=True)
    for (x, y, z) in flips:
        other = flip_bit(other, x, y, z)
    _, flipped = K.permute(other, num_rounds, trace=True, enabled_steps=enabled,
                           round_offset=round_offset)
    return Avalanche(base_trace, flipped, flips)


@dataclass
class Diffusion:
    """For every state bit, the first snapshot at which it depended on the
    chosen input bit (``-1`` = never, within the traced rounds)."""

    source: Tuple[int, int, int]
    first_step: np.ndarray  # (5, 5, 64) int16 snapshot index, -1 if never
    num_rounds: int

    def first_round(self) -> np.ndarray:
        """Snapshot index -> round number (as float: round + step/5), -1 if never."""
        fs = self.first_step.astype(np.float64)
        rounds = np.where(fs <= 0, fs, (fs - 1) / 5.0)
        return np.where(self.first_step < 0, -1.0, rounds)

    def reached_by_round(self, r: int) -> int:
        """How many of the 1600 bits are affected after round r (1-based)."""
        limit = 5 * r
        return int(((self.first_step >= 0) & (self.first_step <= limit)).sum())


def diffusion(
    state: np.ndarray,
    source: Tuple[int, int, int] = (0, 0, 0),
    num_rounds: int = K.NUM_ROUNDS,
    enabled_steps: Iterable[str] = K.STEP_NAMES,
    round_offset: Optional[int] = None,
    base_trace: Optional[K.Trace] = None,
) -> Diffusion:
    """Trace how a single flipped input bit spreads through the state.

    This is a *sample* measure: it flips one bit in the given concrete state
    and reports where the XOR difference first appears.  The linear steps
    (theta, rho, pi) spread the difference deterministically; chi's
    contribution depends on the neighbouring bit values, so the picture is
    data-dependent - exactly what an avalanche test measures."""
    av = avalanche(state, [source], num_rounds, enabled_steps, round_offset, base_trace)
    first = np.full((5, 5, 64), -1, dtype=np.int16)
    for i in range(len(av.base)):
        d = av.diff_bits(i).astype(bool)
        newly = d & (first < 0)
        first[newly] = i
    return Diffusion(source, first, num_rounds)


@dataclass
class BitTrack:
    """One tracked bit followed through a trace: where it is and what value it
    has after every snapshot.  Positions are data-independent (only rho and
    pi move bits); values come from the trace."""

    origin: Tuple[int, int, int]
    positions: List[Tuple[int, int, int]]  # per snapshot
    values: List[int]  # per snapshot
    events: List[str]  # per snapshot, human readable

    def position(self, index: int) -> Tuple[int, int, int]:
        return self.positions[min(index, len(self.positions) - 1)]

    def value(self, index: int) -> int:
        return self.values[min(index, len(self.values) - 1)]

    @property
    def flips(self) -> int:
        return sum(1 for e in self.events if e.startswith("flipped"))


def track_bit(trace: K.Trace, origin: Tuple[int, int, int], bits_fn=None) -> BitTrack:
    """Follow the bit that starts at ``origin`` in snapshot 0 through every step.

    ``bits_fn(index)`` may supply cached (5,5,64) bit arrays per snapshot."""
    if bits_fn is None:
        def bits_fn(i):
            return K.lanes_to_bits(trace[i].state)
    pos = tuple(int(v) for v in origin)
    positions = [pos]
    bits = bits_fn(0)
    values = [int(bits[pos])]
    events = ["start"]
    for snap in trace.snapshots[1:]:
        new_pos = pos if snap.skipped else K.move_cell(snap.step, *pos)
        bits = bits_fn(snap.index)
        v = int(bits[new_pos])
        if snap.skipped:
            ev = "step disabled"
        elif new_pos != pos:
            ev = f"moved to ({new_pos[0]},{new_pos[1]},{new_pos[2]})"
            if snap.step == "rho":
                ev += f"  (z + {int(K.RHO_OFFSETS[pos[0], pos[1]])} mod 64)"
        elif v != values[-1]:
            ev = f"flipped {values[-1]}→{v}"
        else:
            ev = "unchanged"
        pos = new_pos
        positions.append(pos)
        values.append(v)
        events.append(ev)
    return BitTrack(tuple(origin), positions, values, events)


__all__ = ["flip_bit", "Avalanche", "avalanche", "Diffusion", "diffusion", "BitTrack", "track_bit"]
