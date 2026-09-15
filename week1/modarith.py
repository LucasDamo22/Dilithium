"""Week 1, part A: arithmetic in Z_q.

Fill in the functions, then run:  python3 -m pytest test_modarith.py
"""

Q = 8380417  # 2^23 - 2^13 + 1, the ML-DSA modulus


def egcd(a, b):
    """Extended Euclidean algorithm.

    Return (g, x, y) with g = gcd(a, b) and a*x + b*y == g.
    Write the iterative version; recursion is fine too, but iterative is what
    you would build in hardware.
    """
    remainder = 100000000000000000
    if(b != 0 and a != 0):
        if( b >= a):
            dividend = b
            divisor = a
            dividend_x, dividend_y = 0, 1
            divisor_x, divisor_y   = 1, 0
        else:
            dividend = a
            divisor = b
            dividend_x, dividend_y = 1, 0
            divisor_x, divisor_y   = 0, 1
        #gcd
        while remainder > 0:
            quotient = 0

            while quotient * divisor < dividend:
                quotient += 1

            if((quotient * divisor) != dividend):
                quotient -= 1

            remainder = dividend - (quotient*divisor)
            remainder_x = dividend_x - quotient*divisor_x
            remainder_y = dividend_y - quotient*divisor_y
            if remainder == 0:
                break
            dividend = divisor
            divisor = remainder
            dividend_x, dividend_y = divisor_x, divisor_y
            divisor_x, divisor_y = remainder_x, remainder_y
    else:
        if b == 0:
            return (a, 1, 0)
        if a == 0:
            return (b, 0, 1)

    return divisor, divisor_x, divisor_y



def modinv(a, m):
    """Return the inverse of a modulo m: the x in [0, m) with a*x % m == 1.

    Raise ValueError if a is not invertible mod m.
    Build it on egcd -- do not use pow(a, -1, m); the tests use that to check you.
    """
    g, x, y = egcd(a,m)
    if(g != 1):
        raise ValueError
    return x % m
    


def primitive_root_of_unity(order, q):
    """Return some w in Z_q whose multiplicative order is exactly `order`.

    Assume q is prime and `order` is a power of two dividing q - 1.
    Hint: for any a, w = a^((q-1)/order) satisfies w^order == 1, so its order
    divides `order`. For a power of two, how do you check it isn't smaller?
    """
    raise NotImplementedError
