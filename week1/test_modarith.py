import math
import random

import pytest

from modarith import Q, egcd, modinv, primitive_root_of_unity


@pytest.mark.parametrize("a, b", [(240, 46), (17, 7), (7, 17), (12, 18), (5, 0), (Q, 12345)])
def test_egcd_bezout(a, b):
    g, x, y = egcd(a, b)
    assert g == math.gcd(a, b)
    assert a * x + b * y == g


def test_modinv_paper_exercise_1():
    assert modinv(7, 17) == 5


def test_modinv_composite_modulus_when_coprime():
    assert modinv(3, 10) == 7


def test_modinv_not_invertible_raises():
    # paper exercise 2: gcd(4, 10) = 2
    with pytest.raises(ValueError):
        modinv(4, 10)


def test_modinv_random_mod_q():
    for _ in range(1000):
        a = random.randrange(1, Q)
        inv = modinv(a, Q)
        assert 0 <= inv < Q
        assert a * inv % Q == 1
        assert inv == pow(a, -1, Q)


def test_modinv_agrees_with_fermat():
    # paper exercise 3: a^(q-2) is also the inverse when q is prime
    assert modinv(3, 17) == pow(3, 15, 17) == 6
    a = random.randrange(1, Q)
    assert modinv(a, Q) == pow(a, Q - 2, Q)


def test_primitive_8th_root_mod_17():
    w = primitive_root_of_unity(8, 17)
    assert pow(w, 8, 17) == 1
    assert pow(w, 4, 17) == 16  # w^4 = -1, so the order is exactly 8


def test_primitive_512th_root_mod_q():
    w = primitive_root_of_unity(512, Q)
    assert pow(w, 512, Q) == 1
    assert pow(w, 256, Q) == Q - 1  # w^256 = -1, so the order is exactly 512


def test_odd_powers_are_the_roots_of_x256_plus_1():
    # Lesson section 7: w, w^3, ..., w^511 are 256 distinct roots of X^256 + 1,
    # so X^256 + 1 splits into linear factors mod q -- the reason the NTT exists.
    w = primitive_root_of_unity(512, Q)
    roots = {pow(w, 2 * i + 1, Q) for i in range(256)}
    assert len(roots) == 256
    assert all(pow(r, 256, Q) == Q - 1 for r in roots)
