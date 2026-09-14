# Week 1 — Modular arithmetic and polynomial rings

Goal: be fluent in the arithmetic ML-DSA is built on. You need working fluency
here, not abstract algebra. Every section ends with what you actually need to
remember.

**Suggested order**

1. Read this file (about 1-2 h).
2. Do the paper exercises at the bottom *by hand*.
3. Implement `modarith.py`, then run `python3 -m pytest test_modarith.py`.
4. Implement `ring.py`, then run `python3 -m pytest test_ring.py`.

The test cases use the same numbers as the paper exercises, so your
hand-computed answers and your code check each other.

---

## 1. Integers mod q — `Z_q`

`Z_q = {0, 1, ..., q-1}`. You add and multiply as usual, then take the
remainder mod q.

In Python, `%` with a positive modulus always gives a result in `[0, q)`, even
for negative inputs: `-1 % 17 == 16`. **C's `%` does not do this**
(`-1 % 17 == -1`). Remember that when you move to C and hardware.

**Remember:** `a ≡ b (mod q)` means `q` divides `a - b`. Many integers
represent the same element. Section 6 covers choosing which one to use.

## 2. Inverses and the extended Euclidean algorithm

The inverse of `a` mod `m` is the `x` with `a·x ≡ 1 (mod m)`.

**It exists exactly when `gcd(a, m) = 1`.**

The extended Euclidean algorithm computes `g = gcd(a, b)` together with
integers `x, y` such that `a·x + b·y = g` (Bézout's identity). If
`gcd(a, m) = 1`, then `a·x + m·y = 1`. Reduce mod m and the `m·y` term drops
out, leaving `a·x ≡ 1`. So `x` is the inverse.

Worked example, inverse of 7 mod 17:

```
17 = 2·7 + 3
 7 = 2·3 + 1          <- remainder 1, so gcd = 1 and the inverse exists
Back-substitute:
 1 = 7 - 2·3
   = 7 - 2·(17 - 2·7)
   = 5·7 - 2·17
```

So `7⁻¹ ≡ 5 (mod 17)`. Check: `7·5 = 35 = 2·17 + 1`. ✓

When writing code, use the iterative version that tracks `(old_r, r)`,
`(old_x, x)` and `(old_y, y)`. Don't back-substitute.

## 3. Fermat and Euler

**Fermat's little theorem:** if `q` is prime and `a ≢ 0`, then
`a^(q-1) ≡ 1 (mod q)`.

Two consequences you will use:

- A second way to invert: `a⁻¹ ≡ a^(q-2)`, since `a · a^(q-2) = a^(q-1) ≡ 1`.
  With square-and-multiply this takes about log₂ q multiplications.
- **The order of every nonzero element divides `q - 1`.** The order is the
  smallest `k > 0` with `a^k ≡ 1`. This fact is the basis of Section 7.

**Euler's theorem** generalises Fermat to any modulus: if `gcd(a, m) = 1`, then
`a^φ(m) ≡ 1 (mod m)`. Here `φ(m)` counts the integers in `[1, m]` that are
coprime to `m`. For prime `q`, `φ(q) = q - 1`, which gives back Fermat.

## 4. Groups, rings, fields — the vocabulary

| Structure | What it has | Example |
|-----------|-------------|---------|
| **Group** | one operation that is associative, with an identity and inverses | `(Z_q, +)`; also the nonzero elements of `Z_p` under `×` for prime p |
| **Ring**  | `+` and `×`; `+` is a commutative group, `×` is associative and distributes over `+`. Elements do **not** need `×`-inverses | `Z`, `Z_6`, polynomials |
| **Field** | a ring where every nonzero element has a `×`-inverse | `Q`, `R`, `Z_p` for prime p |

**Why `Z_q` is a field exactly when q is prime.** If q is prime, every
`a ∈ [1, q-1]` has `gcd(a, q) = 1`, so it is invertible (Section 2). If
`q = a·b` is composite, then `a·b ≡ 0` with both `a` and `b` nonzero. These are
*zero divisors*, and a zero divisor cannot have an inverse. If `a·x ≡ 1`, then
`b = (a·x)·b = x·(a·b) ≡ 0`, which is a contradiction. Example: in `Z_6`,
`2·3 = 0`, so 2 has no inverse.

One more fact, stated without proof: for prime q, the nonzero elements of `Z_q`
form a **cyclic** group under `×`. Some *generator* `g` has powers
`g, g², ..., g^(q-1)` that hit every nonzero element.

**Remember:** a field is where you can divide. ML-DSA's `q = 8380417` is prime,
so `Z_q` is a field.

## 5. The polynomial ring `R_q = Z_q[X]/(X^n + 1)` — the core of the week

**Elements** are polynomials of degree `< n` with coefficients in `Z_q`:

```
a = a_0 + a_1·X + ... + a_{n-1}·X^{n-1}      stored as the list [a_0, ..., a_{n-1}]
```

For ML-DSA, `n = 256`, so every element is 256 coefficients of 23 bits each.

**Addition** works coefficient by coefficient, mod q. Nothing new.

**Multiplication** takes two steps:

1. Multiply as ordinary polynomials. The result has degree up to `2n - 2`.
2. Reduce using the rule **`X^n = -1`**. Any term at `X^(n+k)` becomes `-X^k`.

In the quotient ring, `X^n + 1` counts as zero, so `X^n = -1`. The resulting
formula for the product `c = a·b` is:

```
c_k =  Σ_{i+j = k}      a_i·b_j        (terms that fit)
     - Σ_{i+j = k + n}  a_i·b_j        (terms that wrapped, with a sign flip)
```

This is **negacyclic convolution**. Compare it with the cyclic convolution you
know from DSP (mod `X^n - 1`), where wrapped terms are added back with no sign
flip.

Worked example: `n = 4`, `q = 17`, so `X^4 = -1`.

```
(1 + 2X + 3X³) · (X + X²)
  = X + X² + 2X² + 2X³ + 3X⁴ + 3X⁵
  = X + 3X² + 2X³ + 3X⁴ + 3X⁵            ordinary product
X⁴ = -1,  X⁵ = X⁴·X = -X:
  = X + 3X² + 2X³ - 3 - 3X
  = -3 - 2X + 3X² + 2X³
  ≡ 14 + 15X + 3X² + 2X³  (mod 17)       -> [14, 15, 3, 2]
```

**The most useful mental picture:** multiplying by `X` shifts every coefficient
up one position. The top coefficient falls off the end and comes back at
position 0 **with its sign flipped**.

```
[a0, a1, a2, a3] · X  =  [-a3, a0, a1, a2]
```

That is why `X^255 · X = X^256 = -1` is the milestone test.

**Why `X^n + 1` and not `X^n - 1`?** `X^n - 1` has the factor `(X - 1)`. That
means "evaluate at X = 1", which sums the coefficients, maps the whole ring
onto plain `Z_q` while preserving `+` and `×`. An attacker could use that map to
shrink the problem. When n is a power of two, `X^n + 1` is irreducible over the
rationals, so there is no shortcut like that. You don't need more detail than
this for now.

**Remember:** negacyclic means wrap around and flip the sign. You will use
this operation more than any other.

## 6. Centred representatives and the infinity norm

Lattice cryptography depends on *small* polynomials. The secret key has
coefficients in `{-2, ..., 2}`, and signatures are rejected when coefficients
get too large. In storage, `-1` is kept as `q - 1`, which looks huge. So
"small" has to be measured in the **centred representative**: the value
congruent to `x` that lies in `(-q/2, q/2]`.

For odd q, that range is `[-(q-1)/2, (q-1)/2]`. With `q = 17`, it is
`[-8, 8]`: `9 → -8` and `16 → -1`.

The **infinity norm** is the largest absolute coefficient, measured in centred
form:

```
‖a‖∞ = max_i |centre(a_i)|
```

The signing rejection check in ML-DSA is essentially `‖z‖∞ < γ₁ - β`. You will
write that line in Week 6.

(FIPS 204 writes the centred reduction as `mod±`.)

## 7. Roots of unity — preview of the NTT

`ω` is a **primitive m-th root of unity** mod q if `ω^m ≡ 1` and `ω^k ≢ 1` for
`0 < k < m`. In other words, `ω` has order exactly `m`.

**It exists if and only if `m` divides `q - 1`.** The multiplicative group is
cyclic of order `q - 1` (Section 4), and a cyclic group has an element of every
order that divides its size. To build one, take any `a` and compute
`ω = a^((q-1)/m)`. By Fermat, `ω^m = a^(q-1) = 1`, so the order of `ω`
*divides* `m`. You then check that it isn't smaller.

**The check is simple when `m` is a power of two.** If `m = 2n`, then `ω` has
order exactly `2n` iff `ω^n ≢ 1`. In fact `ω^n ≡ -1`, because `ω^n` squares to
1 and the only square roots of 1 in a field are `±1`.

**Why ML-DSA needs `2n`, not `n`.** The roots of `X^n + 1` are the values with
`ζ^n = -1`. From the previous paragraph, those are exactly the primitive
2n-th roots `ω, ω³, ω⁵, ..., ω^(2n-1)`. That makes n distinct roots. So when
`2n | q - 1`, `X^n + 1` **splits into n linear factors** mod q. The NTT
evaluates a polynomial at those n points. Multiplication then becomes n
independent scalar multiplications instead of an n² convolution. That is
Week 9. For now you only need the existence condition.

Check it for ML-DSA:

```
q     = 8380417 = 2²³ - 2¹³ + 1
q - 1 = 8380416 = 2¹³ · 3 · 11 · 31
2n    = 512     = 2⁹, which divides 2¹³   ✓
```

The designers chose q this way deliberately: q is prime, fits in 23 bits, has
`q ≡ 1 (mod 512)`, and its sparse binary form makes reduction cheap in
hardware.

---

## Paper exercises

Do these by hand before you write any code.

1. Compute `7⁻¹ mod 17` with the extended Euclidean algorithm. (Section 2 has
   the solution. Do it without looking.)
2. Show that 4 has no inverse mod 10. Find the zero-divisor pair that proves
   it.
3. Compute `3⁻¹ mod 17` as `3^15 mod 17` using square-and-multiply. Check the
   result.
4. In `Z_17[X]/(X^4 + 1)`, compute `(1 + 2X + 3X³)(X + X²)`. (Section 5 has the
   solution. Do it without looking.)
5. In the same ring, compute `X³ · X³`.
6. Show that 2 is a primitive 8th root of unity mod 17. List the four roots of
   `X^4 + 1` mod 17, and check one of them.
7. Give the centred representatives mod 17 of 0, 8, 9 and 16. Find the
   infinity norm of `[14, 15, 3, 2]`.

<details>
<summary>Answers</summary>

1. `5`
2. `gcd(4, 10) = 2 ≠ 1`, and `4·5 = 20 ≡ 0 (mod 10)`.
3. `3² = 9`, `3⁴ = 81 ≡ 13`, `3⁸ ≡ 13² = 169 ≡ 16`.
   `3^15 = 3⁸·3⁴·3²·3 ≡ 16·13·9·3`. Step by step: `16·13 = 208 ≡ 4`,
   `4·9 = 36 ≡ 2`, `2·3 = 6`. Answer `6`, and `3·6 = 18 ≡ 1` ✓.
4. `[14, 15, 3, 2]`
5. `X⁶ = X⁴·X² = -X²` → `[0, 0, 16, 0]`
6. `2⁴ = 16 ≡ -1`, so `2⁸ ≡ 1` and `2⁴ ≢ 1`, which means the order is exactly
   8. The roots are the odd powers: `2, 2³ = 8, 2⁵ ≡ 15, 2⁷ ≡ 9`. Check:
   `8⁴ = 4096 = 240·17 + 16 ≡ -1` ✓.
7. `0, 8, -8, -1`. `[14, 15, 3, 2]` centres to `[-3, -2, 3, 2]`, so the norm is
   `3`.

</details>

---

## Resources

- Modular arithmetic: any discrete maths text. Rosen, *Discrete Mathematics and
  Its Applications*, chapter 4, is enough.
- Groups, rings and fields: the first two chapters of any abstract algebra
  book. Skim for vocabulary, not proofs.
- FIPS 204, Section 2.3 (notation) and Section 2.4 (`mod±`, norms). Read them
  now just to see the notation. Everything else in the standard can wait.
