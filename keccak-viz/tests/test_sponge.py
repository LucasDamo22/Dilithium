"""Sponge, padding, variants: NIST CAVP KATs and hashlib cross-checks."""
import hashlib
import os

import numpy as np
import pytest

from conftest import KAT_DIR, kat_message, parse_rsp
from keccakviz.core import keccak as K
from keccakviz.core import sponge as S

V = S.VARIANTS


# ---------------------------------------------------------------- padding

def test_padding_sha3_shapes():
    assert S.pad(b"", 136, 0x06) == bytes([0x06]) + bytes(134) + bytes([0x80])
    assert S.pad(b"a" * 135, 136, 0x06) == b"a" * 135 + bytes([0x86])
    assert S.pad(b"a" * 136, 136, 0x06)[136:] == bytes([0x06]) + bytes(134) + bytes([0x80])
    assert len(S.pad(b"a" * 137, 136, 0x06)) == 272
    assert S.pad(b"", 168, 0x1F)[0] == 0x1F
    assert S.pad(b"", 136, 0x01) == bytes([0x01]) + bytes(134) + bytes([0x80])
    assert S.padding_bytes(3, 136, 0x06) == bytes([0x06]) + bytes(131) + bytes([0x80])


@pytest.mark.parametrize("n", [0, 1, 135, 136, 137, 271, 272, 273])
def test_padding_length_is_multiple_of_rate(n):
    assert len(S.pad(b"x" * n, 136, 0x06)) % 136 == 0
    assert len(S.pad(b"x" * n, 136, 0x06)) > n


# ---------------------------------------------------------------- known answers

def test_well_known_digests():
    assert S.sha3_256(b"").hex() == "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
    assert S.sha3_256(b"abc").hex() == "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
    assert S.sha3_224(b"").hex() == "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7"
    assert S.sha3_512(b"").hex() == (
        "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6"
        "15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26")
    assert S.shake128(b"", 32).hex() == "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
    # original Keccak-256 (Ethereum's hash)
    assert S.keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert S.keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"


def test_keccak_f_of_zero_state():
    # Keccak-f[1600] applied to the all-zero state (from the Keccak team's test vectors)
    out = K.state_to_bytes(K.permute(K.new_state()))
    assert out[:16].hex() == "e7dde140798f25f18a47c033f9ccd584"


# ---------------------------------------------------------------- hashlib cross-checks

def _msg_cases(rate):
    yield b""
    yield b"\x00"
    yield b"a"
    yield b"abc"
    yield bytes(range(256))[: rate - 1]
    yield bytes(range(256))[:rate]
    yield (bytes(range(256)) * 2)[: rate + 1]
    yield bytes(i % 251 for i in range(3 * rate + 17))
    yield bytes(i % 7 for i in range(1000))


@pytest.mark.parametrize("name", ["sha3_224", "sha3_256", "sha3_384", "sha3_512"])
def test_sha3_matches_hashlib(name):
    v = V[name.upper().replace("_", "-")]
    for m in _msg_cases(v.rate_bytes):
        assert S.digest(m, v) == hashlib.new(name, m).digest(), (name, len(m))


@pytest.mark.parametrize("name", ["shake_128", "shake_256"])
def test_shake_matches_hashlib(name):
    v = V[name.upper().replace("_", "")]
    for m in _msg_cases(v.rate_bytes):
        for n in (0, 1, 16, 32, v.rate_bytes - 1, v.rate_bytes, v.rate_bytes + 1, 3 * v.rate_bytes + 5, 1000):
            assert S.digest(m, v, n) == hashlib.new(name, m).digest(n), (name, len(m), n)


def test_long_shake_output_and_perm_count():
    run = S.sponge(b"hello", V["SHAKE128"], 1000, trace=False)
    assert run.output == hashlib.shake_128(b"hello").digest(1000)
    # 1 absorb block, then ceil(1000/168)=6 squeeze reads -> 5 extra perms
    assert run.num_perm_calls == 1 + 5
    assert len(run.squeeze_blocks) == 6
    run2 = S.sponge(b"x" * 300, V["SHA3-256"], trace=False)
    assert run2.num_perm_calls == 3  # 300 bytes -> 3 blocks of 136
    assert run2.output == hashlib.sha3_256(b"x" * 300).digest()


# ---------------------------------------------------------------- NIST CAVP

def _sha3_files():
    for bits in (224, 256, 384, 512):
        for kind in ("ShortMsg", "LongMsg"):
            yield f"SHA3-{bits}", os.path.join(KAT_DIR, f"SHA3_{bits}{kind}.rsp")


@pytest.mark.parametrize("variant,path", list(_sha3_files()))
def test_nist_cavp_sha3(variant, path):
    entries = parse_rsp(path)
    assert entries, path
    v = V[variant]
    for e in entries:
        m = kat_message(e)
        assert S.digest(m, v).hex() == e["MD"], (variant, e.get("Len"))


def _shake_msg_files():
    for bits in (128, 256):
        for kind in ("ShortMsg", "LongMsg"):
            yield f"SHAKE{bits}", os.path.join(KAT_DIR, f"SHAKE{bits}{kind}.rsp")


@pytest.mark.parametrize("variant,path", list(_shake_msg_files()))
def test_nist_cavp_shake_fixed_output(variant, path):
    entries = parse_rsp(path)
    assert entries, path
    v = V[variant]
    n = 16 if variant == "SHAKE128" else 32
    for e in entries:
        m = kat_message(e)
        assert S.digest(m, v, n).hex() == e["Output"], (variant, e.get("Len"))


@pytest.mark.parametrize("variant", ["SHAKE128", "SHAKE256"])
def test_nist_cavp_shake_variable_output(variant):
    entries = parse_rsp(os.path.join(KAT_DIR, f"{variant}VariableOut.rsp"))
    assert len(entries) > 1000
    v = V[variant]
    for e in entries[::3]:  # every third vector keeps the test quick
        m = bytes.fromhex(e["Msg"])
        n_bits = int(e["Outputlen"])
        assert n_bits % 8 == 0
        assert S.digest(m, v, n_bits // 8).hex() == e["Output"], (variant, e["COUNT"])


# ---------------------------------------------------------------- run record

def test_sponge_run_records_everything():
    m = b"The quick brown fox"
    run = S.sponge(m, V["SHA3-256"])
    assert run.output == hashlib.sha3_256(m).digest()
    assert run.padding[0] == 0x06 and run.padding[-1] == 0x80
    assert len(run.absorb_blocks) == 1 and run.absorb_blocks[0].is_last
    blk = run.absorb_blocks[0]
    assert np.array_equal(blk.state_before, K.new_state())
    assert K.state_to_bytes(blk.state_after_xor)[:136] == blk.data
    assert blk.perm.trace is not None and len(blk.perm.trace) == 121
    assert np.array_equal(blk.perm.trace.initial, blk.state_after_xor)
    assert np.array_equal(blk.perm.trace.final, blk.perm.state_out)
    assert run.squeeze_blocks[0].perm is None
    assert run.timeline() == [("absorb", 0), ("perm", 0), ("squeeze", 0)]


def test_nonstandard_rate_and_reduced_rounds_run():
    v = V["SHA3-256"].with_(name="custom", rate_bytes=100)
    run = S.sponge(b"abc" * 50, v, 40, num_rounds=3)
    assert len(run.output) == 40
    assert run.num_rounds == 3
    assert all(len(c.trace) == 16 for c in run.perm_calls)
    assert run.absorb_blocks[0].perm.trace[5].round_constant == K.ROUND_CONSTANTS[21]


def test_disabling_theta_changes_result_but_keeps_shape():
    run = S.sponge(b"abc", V["SHA3-256"], enabled_steps=("rho", "pi", "chi", "iota"))
    assert run.output != hashlib.sha3_256(b"abc").digest()
    assert run.absorb_blocks[0].perm.trace[1].skipped
