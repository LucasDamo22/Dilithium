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
