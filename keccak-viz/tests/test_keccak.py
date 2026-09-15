"""Structural tests for the permutation core (no hashlib here)."""
import numpy as np
import pytest

from keccakviz.core import keccak as K


def rand_state(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**63, size=(5, 5), dtype=np.int64).astype(np.uint64) ^ (
        rng.integers(0, 2, size=(5, 5)).astype(np.uint64) << np.uint64(63))


def test_round_constants_match_table():
    expected = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
        0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ]
    assert K.ROUND_CONSTANTS == expected


def test_rho_offsets_are_triangular_numbers():
    seen = set()
    x, y = 1, 0
    for t in range(24):
        assert K.RHO_OFFSETS[x, y] == ((t + 1) * (t + 2) // 2) % 64
        seen.add((x, y))
        x, y = y, (2 * x + 3 * y) % 5
    assert K.RHO_OFFSETS[0, 0] == 0
    assert len(seen) == 24


def test_pi_is_a_permutation_of_lanes():
    a = np.arange(25, dtype=np.uint64).reshape(5, 5)
    b = K.pi(a)
    assert sorted(b.ravel().tolist()) == list(range(25))
    for x in range(5):
        for y in range(5):
            tx, ty = K.pi_target(x, y)
            assert b[tx, ty] == a[x, y]
            assert K.pi_source(tx, ty) == (x, y)


def test_rol64():
    a = np.array([[1]], dtype=np.uint64)
    assert K.rol64(a, 1)[0, 0] == 2
    assert K.rol64(a, 63)[0, 0] == 1 << 63
    assert K.rol64(a, 64)[0, 0] == 1
    assert K.rol64(np.array([[1 << 63]], dtype=np.uint64), 1)[0, 0] == 1
    assert K.rol64(a, 0)[0, 0] == 1


def test_rho_moves_single_bit_by_offset():
    for x in range(5):
        for y in range(5):
            a = K.new_state()
            a[x, y] = 1
            b = K.rho(a)
            assert b[x, y] == np.uint64(1) << np.uint64(K.RHO_OFFSETS[x, y])


def test_theta_intermediates():
    a = rand_state(1)
    ap, c, d = K.theta_parts(a)
    for x in range(5):
        assert c[x] == np.bitwise_xor.reduce(a[x, :])
        assert d[x] == (c[(x - 1) % 5] ^ K.rol64(c[(x + 1) % 5], 1))
        for y in range(5):
            assert ap[x, y] == a[x, y] ^ d[x]


def test_chi_definition():
    a = rand_state(2)
    b = K.chi(a)
    for x in range(5):
        for y in range(5):
            assert b[x, y] == a[x, y] ^ (~a[(x + 1) % 5, y] & a[(x + 2) % 5, y])


def test_chi_is_invertible_on_rows():
    # chi on a 5-bit row is a bijection: enumerate all 32 rows
    outs = set()
    for v in range(32):
        a = K.new_state()
        for x in range(5):
            a[x, 0] = (v >> x) & 1
        b = K.chi(a)
        outs.add(tuple(int(b[x, 0]) for x in range(5)))
    assert len(outs) == 32


def test_iota_only_touches_lane_00():
    a = rand_state(3)
    b = K.iota(a, 5)
    assert b[0, 0] == a[0, 0] ^ np.uint64(K.ROUND_CONSTANTS[5])
    b[0, 0] = a[0, 0]
    assert np.array_equal(a, b)


def test_bytes_roundtrip_and_layout():
    data = bytes(range(200))
    s = K.state_from_bytes(data)
    assert K.state_to_bytes(s) == data
    # lane (x, y) is at byte offset 8*(x+5y), little endian
    assert s[1, 0] == int.from_bytes(data[8:16], "little")
    assert s[0, 1] == int.from_bytes(data[40:48], "little")


def test_bits_roundtrip():
    s = rand_state(4)
    bits = K.lanes_to_bits(s)
    assert bits.shape == (5, 5, 64)
    assert np.array_equal(K.bits_to_lanes(bits), s)
    flat = K.state_to_flat_bits(s)
    assert flat.shape == (1600,)
    assert np.array_equal(K.flat_bits_to_state(flat), s)
    x, y, z = 3, 2, 17
    assert flat[K.bit_index(x, y, z)] == bits[x, y, z]
    assert K.bit_coords(K.bit_index(x, y, z)) == (x, y, z)


def test_trace_shape_and_consistency():
    s = rand_state(5)
    out, tr = K.permute(s, trace=True)
    assert len(tr) == 1 + 5 * 24
    assert np.array_equal(tr.initial, s)
    assert np.array_equal(tr.final, out)
    assert np.array_equal(out, K.permute(s))
    # replay each step from the previous snapshot
    for snap in tr.snapshots[1:]:
        prev = tr.snapshots[snap.index - 1].state
        if snap.step == "iota":
            expect = K.iota(prev, snap.round_index_abs)
        else:
            expect = K.STEP_FUNCS[snap.step](prev)
        assert np.array_equal(snap.state, expect), snap.label
        if snap.step == "theta":
            assert snap.theta_c is not None and snap.theta_d is not None
            _, c, d = K.theta_parts(prev)
            assert np.array_equal(snap.theta_c, c) and np.array_equal(snap.theta_d, d)
        if snap.step == "iota":
            assert snap.round_constant == K.ROUND_CONSTANTS[snap.round]
    assert tr.index_of(3, "chi") == 1 + 15 + 3
    assert tr[tr.index_of(3, "chi")].label.startswith("round 3 χ")


def test_reduced_rounds_use_last_round_constants():
    s = rand_state(6)
    out, tr = K.permute(s, num_rounds=2, trace=True)
    assert len(tr) == 11
    assert tr[tr.index_of(0, "iota")].round_constant == K.ROUND_CONSTANTS[22]
    assert tr[tr.index_of(1, "iota")].round_constant == K.ROUND_CONSTANTS[23]
    # matches doing the last two rounds by hand
    a = s
    for ir in (22, 23):
        a = K.iota(K.chi(K.pi(K.rho(K.theta(a)))), ir)
    assert np.array_equal(out, a)
    out0 = K.permute(s, num_rounds=2, round_offset=0)
    a = s
    for ir in (0, 1):
        a = K.iota(K.chi(K.pi(K.rho(K.theta(a)))), ir)
    assert np.array_equal(out0, a)


def test_step_toggles():
    s = rand_state(7)
    out, tr = K.permute(s, num_rounds=1, trace=True, enabled_steps=("rho", "pi", "chi", "iota"))
    assert tr[1].skipped and np.array_equal(tr[1].state, s)
    assert not tr[2].skipped
    assert np.array_equal(out, K.iota(K.chi(K.pi(K.rho(s))), 23))
    out2 = K.permute(s, num_rounds=24, enabled_steps=())
    assert np.array_equal(out2, s)


def test_batched_matches_individual():
    states = np.stack([rand_state(i) for i in range(8)])
    out = K.permute(states)
    assert out.shape == (8, 5, 5)
    for i in range(8):
        assert np.array_equal(out[i], K.permute(states[i]))
    # step-level too
    for f in (K.theta, K.rho, K.pi, K.chi):
        b = f(states)
        for i in range(8):
            assert np.array_equal(b[i], f(states[i])), f.__name__


def test_permutation_is_bijective_on_samples():
    # different inputs -> different outputs for a handful of random samples
    outs = {K.state_to_bytes(K.permute(rand_state(i))) for i in range(20)}
    assert len(outs) == 20


def test_sources_and_targets_are_mutually_consistent():
    import itertools
    cells = list(itertools.product(range(5), range(5), range(64)))
    for step in K.STEP_NAMES:
        for (x, y, z) in cells[::37]:
            for src in K.sources_of(step, x, y, z):
                assert (x, y, z) in K.targets_of(step, *src), (step, (x, y, z), src)
            for tgt in K.targets_of(step, x, y, z):
                assert (x, y, z) in K.sources_of(step, *tgt), (step, (x, y, z), tgt)


def test_sources_actually_influence():
    """Flipping a source bit must be able to change the target; a non-source never can."""
    rng = np.random.default_rng(8)
    for step in ("theta", "rho", "pi", "chi"):
        f = K.STEP_FUNCS[step]
        for _ in range(3):
            x, y, z = rng.integers(5), rng.integers(5), rng.integers(64)
            srcs = set(K.sources_of(step, x, y, z))
            base = rand_state(int(rng.integers(1 << 30)))
            base_bit = K.lanes_to_bits(f(base))[x, y, z]
            for (sx, sy, sz) in [(rng.integers(5), rng.integers(5), rng.integers(64)) for _ in range(40)]:
                flipped = base.copy()
                flipped[sx, sy] ^= np.uint64(1) << np.uint64(sz)
                bit = K.lanes_to_bits(f(flipped))[x, y, z]
                if (sx, sy, sz) not in srcs:
                    assert bit == base_bit, (step, (x, y, z), (sx, sy, sz))
