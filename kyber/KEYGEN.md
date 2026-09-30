# Kyber / ML-KEM — key generation

A detour from the ML-DSA roadmap. The good news: Kyber lives in **the same ring
you already learned in Week 1**, `Z_q[X]/(X^256 + 1)`. Negacyclic wrap-and-flip,
centred representatives, the infinity norm — same furniture. Only the modulus
changes.

| | ML-DSA (Dilithium) | ML-KEM (Kyber) |
|---|---|---|
| `q` | 8380417 | **3329** |
| `n` | 256 | 256 |
| ring | `Z_q[X]/(X^n + 1)` | same |
| purpose | signatures | key encapsulation |

Parameters used throughout this file: **ML-KEM-768**, so `k = 3`, `eta1 = 2`.

Every number below is real output, reproducible with the script at the bottom.

---

## 0. What keygen produces

```
public key   ek = (t_hat, rho)          1184 bytes
secret key   dk = (s_hat, ...)          2400 bytes
```

The secret is **`s`**: a vector of `k = 3` polynomials, each with 256
coefficients. That is 768 numbers, and every single one is one of five values:

```
-2, -1, 0, 1, 2
```

The modulus is 3329, so these 768 numbers *could* have been anywhere in a range
3329 wide, and instead they are all crammed into 5 values.

**Remember:** "the secret is small" is the entire security idea of Kyber.
Section 5 explains why.

---

## 1. One 32-byte seed

Everything — the whole keypair — comes from 32 random bytes:

```
d = 000000000000000000000000000000000000000000000000000000000000002a
```

(fixed here so the numbers reproduce; in reality it comes from the OS entropy
source). **This is the only place true randomness enters keygen.** Everything
after it is deterministic.

## 2. G splits the seed into a public half and a secret half

```
(rho, sigma) = SHA3-512(d || k)

rho   = 5f6d3084fba40975f33eed86498f38f6a24a570c337fb2113a44439aa631e4fb
sigma = 23675228660be9a4232223a006e888c0d495a65aa9cedd3fa025d30b1b722ca8
```

| | job | secrecy |
|---|---|---|
| **rho** | expands into the public matrix `A` | **published** — ships inside the public key |
| **sigma** | expands into `s` and `e` | **never leaves the machine** |

`rho` is publishable because `A` is just a pile of public randomness with no
secret in it. Shipping the 32-byte seed instead of the whole matrix is why
Kyber public keys are small (1184 bytes instead of ~5 KB).

*(Note: the `|| k` domain-separation byte is FIPS 203. Round-3 Kyber hashed
`d` alone.)*

---

## 3. sigma -> the secret `s` and the error `e`

### Stretching the seed

32 bytes cannot produce 768 numbers, so SHAKE-256 stretches it. A **nonce**
keeps each polynomial independent:

```
PRF(sigma, 0) -> s[0]        PRF(sigma, 3) -> e[0]
PRF(sigma, 1) -> s[1]        PRF(sigma, 4) -> e[1]
PRF(sigma, 2) -> s[2]        PRF(sigma, 5) -> e[2]
```

Each call yields `64 * eta1 = 128` bytes of uniformly random garbage. Nothing
small about it yet.

**`s` and `e` are twins.** Same shape, same sampler, same `eta`, same range.
The only difference at birth is the nonce. They diverge only in what happens
afterwards (Section 5).

### The centred binomial distribution (CBD)

Scary name, stupid-simple mechanism. Expand the PRF bytes to bits, **LSB first
within each byte**:

```
b7 18 4c 6c ...  ->  1110 1101 0001 1000 0011 0010 ...
```

Now chew through the bits `2*eta = 4` at a time. For each group of four:

- add the **first 2** bits -> `a`
- add the **last 2** bits -> `b`
- the coefficient is **`a - b`**

That is the whole algorithm. Flip two coins, flip two more, subtract the second
count from the first.

Worked by hand from the bits above:

```
1,1 | 1,0   ->  a=2, b=1  ->   1
1,1 | 0,1   ->  a=2, b=1  ->   1
0,0 | 0,1   ->  a=0, b=1  ->  -1
1,0 | 0,0   ->  a=1, b=0  ->   1
0,0 | 1,1   ->  a=0, b=2  ->  -2
```

Real output, first 16 coefficients of `s[0]`:

```
[1, 1, -1, 1, -2, -1, -2, 0, 0, 0, -1, -2, 1, 0, 1, 2]
```

The first five match. Repeat 256 times per polynomial, six PRF calls total.

There is **no rejection, no retry, no modular reduction** — the construction
makes a large value impossible. That is why it is used: constant-time and
branch-free, which matters for side-channel resistance. Contrast with `A`
(Section 4), which *does* loop.

### Why binomial instead of uniform over [-2, 2]

The shape matters. Counting outcomes of 4 coin flips:

```
-2  ##                    1/16
-1  ########              4/16
 0  ############          6/16
 1  ########              4/16
 2  ##                    1/16
```

A bell curve peaked at 0. Measured over the 256 real coefficients of `s[0]`:

```
        -2    -1     0     1     2
count   17    56    93    61    29
ideal   16    64    96    64    16
```

Two reasons. **Security:** the Module-LWE hardness proofs want an error
distribution approximating a discrete Gaussian; a binomial is the cheapest
usable stand-in. **Practical:** it biases hard toward 0, so `s` is smaller on
average than the `[-2,2]` bound suggests — smaller secret means less noise
growth means fewer decryption failures.

### The eta parameters

| | k | eta1 | eta2 | keygen range |
|---|---|---|---|---|
| ML-KEM-512 | 2 | **3** | 2 | `[-3, 3]` |
| ML-KEM-768 | 3 | 2 | 2 | `[-2, 2]` |
| ML-KEM-1024 | 4 | 2 | 2 | `[-2, 2]` |

Keygen uses **`eta1` for both `s` and `e`**. `eta2` does **not** appear in
keygen at all — it is for the error terms in *encapsulation*. For 768 and 1024
the two happen to be equal, which makes it easy to miss that they are separate
knobs.

---

## 4. rho -> the public matrix `A`

`A` is a `k x k` matrix of polynomials, coefficients **uniform** in `[0, 3329)`.
Each of the `k^2 = 9` cells is generated independently:

```
for i in 0..k-1:
  for j in 0..k-1:
    A_hat[i][j] = SampleNTT(rho || j || i)      # fresh XOF, state reset
```

### 3 bytes -> two 12-bit candidates

Squeeze SHAKE-128 three bytes at a time; 24 bits splits into two 12-bit values:

```
C = [c0, c1, c2]
d1 = c0 + 256 * (c1 mod 16)        # low nibble of c1 + all of c0
d2 = (c1 div 16) + 16 * c2         # high nibble of c1 + all of c2
```

### The critical part: reject, do NOT take mod

12 bits gives `[0, 4095]`. You need `[0, 3328]`.

```
WRONG:   coefficient = value mod 3329
RIGHT:   if value < 3329: keep it.  else: THROW IT AWAY and squeeze again.
```

Taking mod would bias the result: values 0-766 would be hit **twice** (directly,
and by 3329-4095 folding down) while 767-3328 are hit once. `A` would not be
uniform, and the Module-LWE assumption *requires* a uniform `A`.

So `4096 - 3329 = 767` of every 4096 candidates is discarded — an **18.7%**
rejection rate. Measured over the full 3x3 matrix: **513 rejections out of 2817
candidates = 18.2%**.

Real trace, cell (0,0), absorbing `rho || 00 || 00`:

```
  d1 45 bd  ->  d1 =  1489 keep     d2 =  3028 keep
  ea 2d ec  ->  d1 =  3562 REJECT   d2 =  3778 REJECT
  09 8b c2  ->  d1 =  2825 keep     d2 =  3112 keep
  cb 32 7e  ->  d1 =   715 keep     d2 =  2019 keep
  ed b5 68  ->  d1 =  1517 keep     d2 =  1675 keep
  9b dc 7e  ->  d1 =  3227 keep     d2 =  2029 keep
  a5 aa b8  ->  d1 =  2725 keep     d2 =  2954 keep
  ...

A_hat[0][0] first 12 : [1489, 3028, 2825, 3112, 715, 2019, 1517, 1675,
                        3227, 2029, 2725, 2954]
bytes consumed: 483   rejections: 65
```

Average across the matrix: **470 bytes per cell**, i.e. 2.8 SHAKE-128 blocks
(the block is 168 bytes), so most cells squeeze 3 blocks and some need 4.

Because the byte count is data-dependent, this is the **only loop in Kyber that
runs a variable number of times**. It is safe only because it depends on `rho`,
which is public. Rejection sampling anywhere near `sigma` would be a timing
leak.

### Gotcha: the index bytes are swapped

For the cell at row `i`, column `j`, the absorbed seed is

```
rho || j || i        <- COLUMN first, then row
```

So cell `(0,1)` appends bytes `01 00`, not `00 01`. Get this backwards and you
build the transpose: all your own self-tests pass and nothing interoperates.

### Gotcha: the output is already `A_hat`, not `A`

The matrix is never built in the normal domain and transformed. The sampled
bytes **are** the NTT representation, by declaration.

This is legitimate because the NTT is a bijection: if `A` is uniform, so is
`NTT(A)`. "Sample uniform and call it `A_hat`" and "sample uniform, call it
`A`, then transform" give identical distributions. Kyber skips the transform
and saves `k^2 = 9` NTTs for free. That is why the spec writes the hat from
the first line.

---

## 5. Putting it together

```
s_hat = NTT(s)
e_hat = NTT(e)
t_hat = A_hat o s_hat + e_hat          # o = pointwise multiply in NTT domain
```

An attacker sees `A` (recomputable from public `rho`) and `t`, and wants `s`.

Without `e`, that is `s = A^-1 t` — a linear system, solved in seconds. The
error `e` destroys that: the problem becomes "find the `s` making `A*s`
*nearly* equal `t`", which is Module-LWE, and nobody knows how to do it.

But it only works because both are **small**:

- If `s` and `e` were full-size random mod 3329, `t` would be pure noise. Every
  possible `s` would have some `e` explaining it, so `t` would carry no
  information about `s` — unbreakable, but also undecryptable.
- Small is what makes `s` simultaneously **hidden from an attacker** and
  **recoverable by the owner**.

That is the tightrope the whole scheme walks.

### What is kept and what is thrown away

| | fate |
|---|---|
| `s_hat` | stored in the secret key, used forever to decrypt |
| `e` | used **once** to compute `t`, then **deleted** — never stored |
| `rho` | published in the public key |
| `sigma` | discarded after `s` and `e` are derived |

`e` is scaffolding. Its life's work is to be added once.

`s` is stored **already in NTT form** because every later operation needs it
that way — pay for the transform once at keygen, never again.

### Sizes

```
t_hat : 256 coeffs * 12 bits = 384 bytes per polynomial, * k = 1152
ek    : 1152 + 32 (rho)      = 1184 bytes
dk    : 768k + 96            = 2400 bytes
```

---

## Gotcha checklist

1. `A` uses **rejection**, not `mod q`. Taking mod biases the matrix and breaks
   the security argument.
2. Index order is `rho || j || i` — **column then row**.
3. What you sample for `A` is **already in the NTT domain**.
4. CBD bits are read **LSB first** within each byte.
5. Keygen uses `eta1` for **both** `s` and `e`. `eta2` belongs to encapsulation.
6. `eta1 = 3` for ML-KEM-512, giving range `[-3, 3]`, not `[-2, 2]`.
7. `e` is never stored. `sigma` is never stored.

---

## Reproduce every number in this file

```python
import hashlib, collections

Q, N, K, ETA1 = 3329, 256, 3, 2                      # ML-KEM-768

d = bytes.fromhex('00'*31 + '2a')
h = hashlib.sha3_512(d + bytes([K])).digest()
rho, sigma = h[:32], h[32:]

def bytes_to_bits(bs):                                # LSB first
    return [(b >> i) & 1 for b in bs for i in range(8)]

def cbd(buf, eta):                                    # s and e
    bits = bytes_to_bits(buf)
    out = []
    for i in range(N):
        a = sum(bits[2*i*eta + j] for j in range(eta))
        b = sum(bits[2*i*eta + eta + j] for j in range(eta))
        out.append(a - b)                             # centred, in [-eta, eta]
    return out

def sample_ntt(seed, i, j):                           # one cell of A
    buf = hashlib.shake_128(seed + bytes([j, i])).digest(3 * 400)
    a, p, rej = [], 0, 0
    while len(a) < 256:
        c0, c1, c2 = buf[p], buf[p+1], buf[p+2]; p += 3
        d1 = c0 + 256 * (c1 % 16)
        d2 = (c1 // 16) + 16 * c2
        for v in (d1, d2):
            if v < Q and len(a) < 256: a.append(v)
            elif v >= Q: rej += 1
    return a, rej, p

s = [cbd(hashlib.shake_256(sigma + bytes([n])).digest(64*ETA1), ETA1) for n in range(K)]
e = [cbd(hashlib.shake_256(sigma + bytes([n])).digest(64*ETA1), ETA1) for n in range(K, 2*K)]

print('rho  ', rho.hex())
print('s[0] ', s[0][:16])
print('dist ', dict(sorted(collections.Counter(s[0]).items())))
print('e[0] ', e[0][:12])
a00, rej, used = sample_ntt(rho, 0, 0)
print('A[0][0]', a00[:12], '| bytes', used, '| rejections', rej)
```

---

## Next

- Encapsulation: `r`, `e1`, `e2` (this is where `eta2` shows up), compression,
  and the Fujisaki-Okamoto transform that turns the PKE into a real KEM.
- Decapsulation and the implicit-rejection trick.
