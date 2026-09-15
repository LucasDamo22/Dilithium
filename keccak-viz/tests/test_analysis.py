import json

import numpy as np

from keccakviz.core import analysis as A
from keccakviz.core import export as E
from keccakviz.core import keccak as K
from keccakviz.core import layouts as L
from keccakviz.core import sponge as S


def test_avalanche_reaches_half_state():
    s = K.new_state()
    av = A.avalanche(s, [(0, 0, 0)])
    h = av.hamming_per_round()
    assert h[0] == 1
    assert h[1] < 50  # after one round: still local
    assert 700 <= h[4] <= 900  # full diffusion after a few rounds
    assert all(700 <= v <= 900 for v in h[6:])


def test_avalanche_without_theta_is_crippled():
    s = K.new_state()
    av = A.avalanche(s, [(0, 0, 0)], enabled_steps=("rho", "pi", "chi", "iota"))
    h = av.hamming_per_round()
    # without theta a single-bit difference spreads only via chi in its own row,
    # so growth is far slower than with theta
    assert h[3] < 100
    full = A.avalanche(s, [(0, 0, 0)]).hamming_per_round()
    assert full[3] > 4 * h[3]


def test_diffusion_first_step_is_monotone_and_starts_at_source():
    s = K.state_from_bytes(bytes(range(200)))
    d = A.diffusion(s, (1, 2, 3))
    assert d.first_step[1, 2, 3] == 0
    assert (d.first_step == 0).sum() == 1
    # theta spreads the flipped bit to 11 cells (itself + 2 columns) after step 1
    assert (d.first_step <= 1).sum() == 11
    assert d.reached_by_round(d.num_rounds) == 1600
    fr = d.first_round()
    assert fr[1, 2, 3] == 0 and fr.max() < d.num_rounds


def test_interleave_roundtrip_and_rotation():
    rng = np.random.default_rng(1)
    for _ in range(50):
        lane = int(rng.integers(0, 2**63)) | (int(rng.integers(0, 2)) << 63)
        e, o = L.interleave(lane)
        assert L.deinterleave(e, o) == lane
        for r in (0, 1, 2, 3, 31, 32, 33, 62, 63):
            expect = int(K.rol64(np.array([[lane]], dtype=np.uint64), r)[0, 0])
            assert L.deinterleave(*L.rol64_interleaved(e, o, r)) == expect, r


def test_state_interleaved_matches_scalar():
    s = K.state_from_bytes(bytes(range(200)))
    even, odd = L.state_interleaved(s)
    for x in range(5):
        for y in range(5):
            e, o = L.interleave(int(s[x, y]))
            assert int(even[x, y]) == e and int(odd[x, y]) == o


def test_batched_theta_parity_is_elementwise():
    states = [K.state_from_bytes(bytes((i * 7 + j) & 0xFF for j in range(200))) for i in range(4)]
    b = L.batch_states(states)
    assert b.shape == (4, 5, 5)
    assert L.theta_parity_is_elementwise(b)
    assert np.array_equal(L.register(b, 2, 3), np.array([s[2, 3] for s in states]))
    out = K.permute(b)
    for i, s in enumerate(states):
        assert np.array_equal(out[i], K.permute(s))


def test_json_export_roundtrips_state():
    run = S.sponge(b"abc", S.VARIANTS["SHAKE128"], 200)
    d = json.loads(E.run_to_json(run))
    assert d["output_hex"] == run.output.hex()
    assert len(d["perm_calls"]) == run.num_perm_calls
    tr = d["perm_calls"][0]["trace"]
    assert len(tr["snapshots"]) == 121
    snap = tr["snapshots"][1]
    assert snap["step"] == "theta" and len(snap["theta_C"]) == 5
    last = tr["snapshots"][-1]
    assert bytes.fromhex(last["state_bytes_hex"]) == K.state_to_bytes(run.perm_calls[0].state_out)
    assert int(last["lanes_hex_xy"][1][0], 16) == int(run.perm_calls[0].state_out[1, 0])


def test_track_bit_follows_rho_and_pi():
    s = K.state_from_bytes(bytes(range(200)))
    _, tr = K.permute(s, num_rounds=2, trace=True)
    t = A.track_bit(tr, (1, 0, 5))
    assert len(t.positions) == len(tr) == 11
    assert t.positions[0] == (1, 0, 5)
    assert t.positions[1] == (1, 0, 5)  # theta does not move
    assert t.positions[2] == (1, 0, (5 + int(K.RHO_OFFSETS[1, 0])) % 64)  # rho
    x, y, z = t.positions[2]
    assert t.positions[3] == (K.pi_target(x, y) + (z,))  # pi
    assert t.positions[5] == t.positions[3]  # chi, iota stay
    # values match the trace at the tracked position
    for k, snap in enumerate(tr.snapshots):
        assert t.values[k] == int(K.lanes_to_bits(snap.state)[t.positions[k]])
    assert t.events[2].startswith("moved to") and t.events[3].startswith("moved to")
    # a moved bit keeps its value
    assert t.values[2] == t.values[1] and t.values[3] == t.values[2]
    assert all(e in ("unchanged",) or e.startswith("flipped") for e in (t.events[1], t.events[4], t.events[5]))


def test_track_bit_with_disabled_rho_stays_put():
    s = K.state_from_bytes(bytes(range(200)))
    _, tr = K.permute(s, num_rounds=1, trace=True, enabled_steps=("theta", "pi", "chi", "iota"))
    t = A.track_bit(tr, (2, 3, 7))
    assert t.positions[2] == (2, 3, 7) and t.events[2] == "step disabled"


def test_dye_is_conserved_and_becomes_homogeneous():
    s = K.state_from_bytes(bytes(range(200)))
    _, tr = K.permute(s, trace=True)
    init = np.zeros((1600, 2), dtype=np.float32)
    init[0, 0] = 1.0  # red on cell (0,0,0)
    init[320 * 3 + 64 * 2 + 17, 1] = 1.0  # blue on cell (3,2,17); flat order is 320x + 64y + z
    d = A.propagate_dye(tr, init)
    assert d.shape == (121, 1600, 2)
    tot0 = d[0].sum(axis=0)
    for k in (1, 5, 60, 120):
        assert np.allclose(d[k].sum(axis=0), tot0, atol=1e-3)  # conserved
    assert A.dye_spread(d[0]) == 800.0  # two dyed cells: max 1 vs mean 2/1600
    assert A.dye_spread(d[5]) < 200
    assert abs(A.dye_spread(d[120]) - 1.0) < 1e-3  # homogeneous at the end
    assert np.allclose(d[120], d[120].mean(axis=0), atol=1e-4)
    # a single theta spreads a dye to 11 cells, chi to 3
    tbl = A.source_table("theta")
    assert tbl.shape == (1600, 11) and A.source_table("chi").shape == (1600, 3)
    assert A.source_table("rho").shape == (1600, 1)


def test_load_snapshots_and_full_trace():
    run = S.sponge(b"x" * 300, S.VARIANTS["SHA3-256"])
    blk = run.absorb_blocks[1]  # second block: the state before is non-zero
    snaps = S.load_snapshots(blk.state_before, blk.data, word_bits=32, words_per_cycle=2)
    n_words = 136 * 8 // 32
    assert snaps[0].info["phase"] == "seed" and snaps[1].info["phase"] == "iv"
    assert K.state_to_bytes(snaps[0].state)[:136] == blk.data
    assert np.array_equal(snaps[1].state, blk.state_before)
    assert len(snaps) == 2 + n_words // 2
    assert np.array_equal(snaps[-1].state, blk.state_after_xor)  # everything loaded == absorb XOR
    # each cycle touches exactly its 2 words = 64 bit positions
    assert len(snaps[2].info["cells"]) == 64 and snaps[2].info["words"] == [0, 1]
    ft = S.full_trace(blk.perm, blk, 32, 2)
    assert ft.n_load == len(snaps)
    assert len(ft) == len(snaps) + 121
    assert ft[ft.n_load].step == "initial" and ft.core_index(ft.n_load + 7) == 7
    assert ft.index_of(0, "theta") == ft.n_load + 1
    assert ft.round_end_indices()[-1] == len(ft) - 1
    # tracking through the loading phase: no movement, a "loaded" bit may flip
    t = A.track_bit(ft, (0, 0, 0))
    assert len(t.positions) == len(ft) and all(p == (0, 0, 0) for p in t.positions[: ft.n_load])
    # squeeze call: no loading
    run2 = S.sponge(b"hi", S.VARIANTS["SHAKE128"], 400)
    ft2 = S.full_trace(run2.perm_calls[1], None)
    assert ft2.n_load == 0 and len(ft2) == 121
    # full-width bus: one cycle
    one = S.load_snapshots(blk.state_before, blk.data, 64, 17)
    assert len(one) == 3
