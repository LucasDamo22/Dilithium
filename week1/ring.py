"""Week 1, part B: the ring R_q = Z_q[X]/(X^n + 1).

A polynomial a_0 + a_1*X + ... + a_{n-1}*X^{n-1} is a plain Python list of n
ints, index i holding the coefficient of X^i, each in [0, q).

n is taken from len(a), and q is a parameter defaulting to the ML-DSA modulus,
so the same code runs on the toy ring from the lesson (n = 4, q = 17) and the
real one (n = 256, q = 8380417).

Plain integers only -- no NumPy. Correctness and understanding, not speed.
Fill in the functions, then run:  python3 -m pytest test_ring.py
"""

N = 256
Q = 8380417


def poly_add(a, b, q=Q):
    """Coefficient-wise a + b mod q. Return a new list; do not modify inputs."""
    raise NotImplementedError


def poly_sub(a, b, q=Q):
    """Coefficient-wise a - b mod q. Results must land in [0, q)."""
    raise NotImplementedError


def poly_mul(a, b, q=Q):
    """Schoolbook product of a and b in Z_q[X]/(X^n + 1).

    Every pair (i, j) contributes a[i]*b[j] to position i + j. When i + j >= n,
    use X^n = -1: the term lands on position i + j - n with its sign flipped.
    """
    raise NotImplementedError


def center(x, q=Q):
    """Centred representative of x mod q: the integer congruent to x that lies
    in (-q/2, q/2]. x may be any integer, including negative or >= q.
    """
    raise NotImplementedError


def infinity_norm(a, q=Q):
    """max |center(a_i)| over all coefficients. 0 for the zero polynomial."""
    raise NotImplementedError
