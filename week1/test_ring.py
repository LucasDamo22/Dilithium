import random

import pytest

from ring import N, Q, center, infinity_norm, poly_add, poly_mul, poly_sub


def monomial(k, coeff=1, n=N, q=Q):
    """coeff * X^k as a coefficient list."""
    p = [0] * n
    p[k] = coeff % q
    return p


def rand_poly(n=N, q=Q):
    return [random.randrange(q) for _ in range(n)]


ZERO = [0] * N


# --- addition and subtraction ------------------------------------------------

def test_add_wraps_mod_q():
    assert poly_add(monomial(0, Q - 1), monomial(0, 5)) == monomial(0, 4)


def test_sub_wraps_mod_q():
    assert poly_sub(ZERO, monomial(3)) == monomial(3, -1)


def test_add_sub_roundtrip():
    a, b = rand_poly(), rand_poly()
    assert poly_sub(poly_add(a, b), b) == a


def test_add_does_not_modify_inputs():
    a, b = rand_poly(), rand_poly()
    a_copy, b_copy = a[:], b[:]
    poly_add(a, b)
    poly_sub(a, b)
    assert a == a_copy and b == b_copy


# --- multiplication: toy ring from the lesson --------------------------------

def test_mul_toy_paper_exercise_4():
    # (1 + 2X + 3X^3)(X + X^2) in Z_17[X]/(X^4 + 1)
    assert poly_mul([1, 2, 0, 3], [0, 1, 1, 0], q=17) == [14, 15, 3, 2]


def test_mul_toy_paper_exercise_5():
    # X^3 * X^3 = X^6 = X^4 * X^2 = -X^2
    assert poly_mul([0, 0, 0, 1], [0, 0, 0, 1], q=17) == [0, 0, 16, 0]


def test_mul_toy_x_to_the_n_is_minus_one():
    n, q = 8, 17
    x = monomial(1, n=n, q=q)
    p = monomial(0, n=n, q=q)
    for _ in range(n):
        p = poly_mul(p, x, q=q)
    assert p == monomial(0, -1, n=n, q=q)


# --- multiplication: the real ring -------------------------------------------

def test_x255_times_x_is_minus_one():
    # The roadmap milestone.
    assert poly_mul(monomial(255), monomial(1)) == monomial(0, -1)


def test_x255_squared():
    # X^510 = X^256 * X^254 = -X^254
    assert poly_mul(monomial(255), monomial(255)) == monomial(254, -1)


def test_mul_by_one():
    a = rand_poly()
    assert poly_mul(a, monomial(0)) == a


def test_mul_by_x_is_negacyclic_shift():
    # shift up one; the top coefficient wraps to position 0 with its sign flipped
    a = rand_poly()
    assert poly_mul(a, monomial(1)) == [(-a[-1]) % Q] + a[:-1]


def test_mul_commutative():
    a, b = rand_poly(), rand_poly()
    assert poly_mul(a, b) == poly_mul(b, a)


def test_mul_associative():
    a, b, c = rand_poly(), rand_poly(), rand_poly()
    assert poly_mul(poly_mul(a, b), c) == poly_mul(a, poly_mul(b, c))


def test_mul_distributes_over_add():
    a, b, c = rand_poly(), rand_poly(), rand_poly()
    assert poly_mul(a, poly_add(b, c)) == poly_add(poly_mul(a, b), poly_mul(a, c))


def test_mul_output_in_range():
    assert all(0 <= x < Q for x in poly_mul(rand_poly(), rand_poly()))


# --- centred representatives and the infinity norm ---------------------------

@pytest.mark.parametrize("x, expected", [(0, 0), (8, 8), (9, -8), (16, -1)])
def test_center_toy_paper_exercise_7(x, expected):
    assert center(x, q=17) == expected


def test_center_boundaries():
    half = (Q - 1) // 2
    assert center(half) == half
    assert center(half + 1) == -half
    assert center(Q - 1) == -1


def test_center_accepts_any_integer():
    assert center(-1) == -1
    assert center(Q + 3) == 3
    assert center(-Q - 3) == -3


def test_center_range_and_congruence():
    for _ in range(1000):
        x = random.randrange(-10 * Q, 10 * Q)
        c = center(x)
        assert -Q / 2 < c <= Q / 2
        assert (c - x) % Q == 0


def test_infinity_norm_toy_paper_exercise_7():
    assert infinity_norm([14, 15, 3, 2], q=17) == 3


def test_infinity_norm_zero():
    assert infinity_norm(ZERO) == 0


def test_infinity_norm_counts_negatives():
    # -5 is stored as Q - 5, which looks huge but is small
    assert infinity_norm(monomial(7, -5)) == 5


def test_infinity_norm_max_possible():
    assert infinity_norm(monomial(0, (Q - 1) // 2)) == (Q - 1) // 2
