# Roadmap: from zero cryptography to ML-DSA (Dilithium)

Target: understand vanilla ML-DSA well enough to implement it from the standard
and explain every design decision. Vector acceleration comes *after* this.

Assumed background: computer engineering, comfortable with C and Python, linear
algebra, probability and statistics. No cryptography.

Estimated effort: **10 weeks part-time** (~6-8 h/week). Compressible to 6 weeks
if you skip the optional depth.

---

## The one structural insight

**Dilithium is a lattice Schnorr signature.**

| Schnorr | Dilithium |
|---------|-----------|
| commit to random `k` | commit to random `y` |
| challenge `c` | challenge `c` |
| response `s = k + c·x` | response `z = y + c·s₁` |
| `x` is the secret scalar | `s₁` is the secret polynomial vector |

Learn Schnorr first (Phase 3). Everything after it becomes "Schnorr, but over
polynomial rings, plus a rejection step so the response does not leak."

Most people go straight at FIPS 204 and bounce off. The Schnorr detour costs a
week and saves a month.

---

## Working method

**Implement as you read.** You are a hardware person; you learn by building.
Every phase has a build milestone. Write everything in Python with plain
integers first — no NumPy, no optimization. Correctness and understanding only.

Keep one repository, `dilithium-from-scratch`, with a directory per phase. By
the end you will have a readable reference implementation that you fully
understand, which is worth more than any set of notes.

---

## Phase 1 — Modular arithmetic and polynomial rings (Week 1)

You need working fluency, not a full abstract algebra course.

### Concepts

- Integers mod q: `Z_q`. Addition, multiplication, modular inverse.
- Extended Euclidean algorithm; when inverses exist.
- Fermat's little theorem, Euler's theorem.
- Groups, rings, fields — just enough to know what the words mean and why
  `Z_q` is a field exactly when q is prime.
- **Polynomial rings**: `R_q = Z_q[X]/(X^n + 1)`.
  The key fact: `X^n = -1`, so when a product overflows degree n, the
  coefficients **wrap around with a sign flip**. This is called *negacyclic*
  convolution. Get this in your fingers — it is the single most-used operation
  in the whole scheme.
- Roots of unity in a finite field. Why `q ≡ 1 (mod 2n)` guarantees a primitive
  2n-th root of unity exists, and why that is exactly the condition for the NTT.

### Build milestone

A Python module implementing `R_q` for n = 256, q = 8380417:

- `poly_add`, `poly_sub`
- `poly_mul` by naive schoolbook convolution with negacyclic reduction
- centred representative reduction (map to `(-q/2, q/2]`)
- `infinity_norm`

Test: `(X^255) * (X) == -1` in your ring. If that works, you have negacyclic
reduction right.

### Resources

- Any discrete mathematics text for modular arithmetic.
- For rings: the first two chapters of almost any abstract algebra book. Skim.
  You need vocabulary, not proofs.

---

## Phase 2 — Cryptographic foundations (Week 2)

### Concepts

- Symmetric vs asymmetric cryptography.
- **Hash functions** and their three properties: preimage resistance, second
  preimage resistance, collision resistance.
- SHA-3 and SHAKE. SHAKE is an *extendable output function* (XOF) — you request
  as many output bytes as you want. Dilithium uses this constantly.
- The **random oracle model** — the idealization used in security proofs. You
  need to know what it means when a paper says "in the ROM."
- **Digital signature definition**: KeyGen, Sign, Verify.
- **EUF-CMA**: existential unforgeability under chosen-message attack. Note it
  is stronger than "cannot recover the key" — forging one signature is a break.
- Security levels: what "128-bit security" actually claims.

### Build milestone

Wrap SHAKE128 and SHAKE256 from Python's `hashlib` into a small XOF class with a
`squeeze(nbytes)` method. Trivial code, but you will use it in every later phase
and it forces you to understand the absorb/squeeze model.

### Resources

- **Boneh & Shoup, *A Graduate Course in Applied Cryptography*** — free online.
  Chapters on hash functions and digital signatures. Read selectively.
- **Katz & Lindell, *Introduction to Modern Cryptography*** — the standard
  textbook if you want a more structured treatment.
- Dan Boneh's Cryptography I course (Coursera / Stanford) if you prefer video.

---

## Phase 3 — Schnorr and Fiat-Shamir (Week 3) — THE PIVOT

**Do not skip or compress this phase.** It is what makes Dilithium legible.

### Concepts

- **Identification protocols**: proving you know a secret without revealing it.
- **Sigma protocols**: the three-move structure — commitment, challenge,
  response.
- **Schnorr identification** over a group of prime order:
  - Prover picks random `k`, sends `R = g^k`.
  - Verifier sends challenge `c`.
  - Prover sends `s = k + c·x`.
  - Verifier checks `g^s == R · y^c` where `y = g^x`.
- **Honest-verifier zero-knowledge**: why `s` reveals nothing about `x` —
  because `k` is uniform and masks it perfectly.
- **The Fiat-Shamir transform**: replace the verifier's random challenge with
  `c = H(commitment ‖ message)`. The interactive protocol becomes a
  non-interactive signature.
- **Schnorr signatures**, and why nonce reuse is catastrophic (the Sony PS3 /
  ECDSA nonce disaster is the famous illustration — same failure mode).

### Build milestone

Implement Schnorr signatures in Python over a small prime-order group. Sign,
verify, and then **deliberately reuse a nonce across two signatures and recover
the private key from the two outputs.** Doing this by hand is what makes the
masking argument concrete, and it is exactly the intuition you need for
Dilithium's rejection step.

### Resources

- Boneh & Shoup, the chapter on identification protocols and Fiat-Shamir.
- Schnorr's original 1989/1991 papers are short and readable.

---

## Phase 4 — Lattices, SIS and LWE (Weeks 4-5)

### Concepts

**Lattices**
- Definition: all *integer* combinations of a set of basis vectors.
- Good bases vs bad bases. Same lattice, wildly different usability.
- **SVP** (shortest vector problem), **CVP** (closest vector problem).
- LLL basis reduction — conceptually only. You need to know it exists, roughly
  what it achieves, and that it is why lattice dimensions must be large.

**The two hard problems**
- **SIS** (Short Integer Solution): given random `A`, find a short nonzero `z`
  with `Az = 0 mod q`. → this is what makes forgery hard.
- **LWE** (Learning With Errors): given `A` and `t = As + e` with `s`, `e`
  small, recover `s`. → this is what makes key recovery hard.
- **The intuition**: without `e`, LWE is Gaussian elimination — trivial. The
  small error destroys that, because noise compounds catastrophically through
  elimination.

**Structured variants**
- **Ring-LWE**: work in `R_q` instead of `Z_q`. Massively smaller keys, NTT
  makes multiplication fast. But more algebraic structure to attack.
- **Module-LWE**: matrices and vectors *over* the ring. The compromise
  Dilithium uses.
- Why Module-LWE lets you scale security by changing `(k, l)` while keeping
  n = 256 fixed. **This is why one NTT engine serves all three ML-DSA parameter
  sets** — a fact you will care about a lot later.

### Build milestone

Implement toy LWE in Python at small parameters (n = 10, q = 97):
- Generate `A`, small `s`, small `e`; compute `t = As + e`.
- Try to recover `s` by Gaussian elimination and watch it fail.
- Then implement toy Module-LWE with polynomial entries using your Phase 1 ring.

### Resources

- **Peikert, *A Decade of Lattice Cryptography*** (eprint 2015/939) — free,
  and the best single survey. Read chapters 1-4.
- **Micciancio & Regev, "Lattice-based Cryptography"** chapter in *Post-Quantum
  Cryptography* (Bernstein, Buchmann, Dahmen, eds.).
- Regev's original LWE paper (2005) if you want the source.

---

## Phase 5 — Lyubashevsky's construction (Week 6)

The bridge between Phase 3 and Phase 7.

### Concepts

- Why the naive lattice Schnorr **leaks the secret**: in `z = y + c·s₁`, if `y`
  comes from a bounded range, the distribution of `z` depends on `s₁`. Collect
  enough signatures, average, recover the key.
- **Rejection sampling** as the fix: accept `z` only when it falls in a region
  where the shifted and unshifted distributions are *identical*. The published
  `z` is then provably independent of the secret.
- This is why signing takes a variable number of attempts (typically 3-7).
- The cost: variable-time signing, which you will have to address later in any
  hardware paper.

### Build milestone

Take your Phase 3 Schnorr code. Replace the group with your polynomial ring.
Add rejection sampling. Measure the acceptance rate empirically and compare it
to the theoretical prediction. You now have proto-Dilithium.

### Resources

- **Lyubashevsky, "Fiat-Shamir with Aborts," ASIACRYPT 2009.** The origin of
  the idea. Short.
- **Lyubashevsky, "Lattice Signatures Without Trapdoors," EUROCRYPT 2012.** The
  direct ancestor of Dilithium.

---

## Phase 6 — Why post-quantum at all (half a week, anywhere)

Short phase, read whenever you want a break.

- **Shor's algorithm**: solves factoring and discrete log in polynomial time.
  RSA and ECDSA are not weakened, they are broken.
- **Grover's algorithm**: only a quadratic speedup, so symmetric crypto and
  hashes survive with adjusted parameters. This asymmetry is why Dilithium can
  lean on SHAKE freely.
- The NIST PQC standardization process and its outcome: ML-KEM (FIPS 203),
  ML-DSA (FIPS 204), SLH-DSA (FIPS 205), and later HQC.
- The families: lattice, hash-based, code-based, multivariate, isogeny — and
  roughly why lattices won on balance of size and speed.
- "Harvest now, decrypt later" and why migration is happening before quantum
  computers exist.

---

## Phase 7 — ML-DSA proper (Weeks 7-8)

Now FIPS 204 will read easily.

### Read in this order

1. The **CRYSTALS-Dilithium round 3 specification** (pq-crystals.org) *first*.
   It is more expository than the standard — it explains *why*, not just *what*.
2. Then **FIPS 204**. Precise, complete pseudocode, and the normative version.
   Note it differs from round-3 Dilithium in some details (domain separation,
   the `tr` computation, hedged vs deterministic signing).

### Concepts to nail

**Parameters** — know what each one controls:
`n = 256`, `q = 8380417 = 2^23 - 2^13 + 1`, `d = 13`, and per-set `(k, l)`,
`eta`, `tau`, `gamma1`, `gamma2`, `beta`, `omega`.

Understand *why* q has that form: `q ≡ 1 (mod 512)` gives the 512th roots of
unity needed for the negacyclic NTT.

**The supporting functions.** These are where people get lost. Work each one on
paper with small numbers before coding:
- `Power2Round` — split `t` into high and low bits, enabling the small public key
- `Decompose`, `HighBits`, `LowBits`
- `MakeHint`, `UseHint` — the clever part. The hint is a handful of bits that
  lets the verifier reconstruct `w₁` despite not knowing `t₀`.

**The sampling functions**, all SHAKE-driven:
- `ExpandA` — generates the whole matrix from a 32-byte seed by rejection
  sampling. A is never stored.
- `ExpandS`, `ExpandMask`
- `SampleInBall` — the challenge: a sparse polynomial with exactly `tau`
  coefficients in `{-1, +1}`.

**Encoding and packing.** Tedious, bit-level, and a common source of bugs. Do
not skip; the test vectors will not pass without byte-exact packing.

**Deterministic vs hedged signing**, and the `mu` / `tr` domain separation.

### Build milestone

Implement ML-DSA-44 KeyGen, Sign and Verify in Python, using naive schoolbook
polynomial multiplication (no NTT yet). Self-consistency test: sign 1000 random
messages, verify all, then flip one bit of each signature and confirm all fail.

---

## Phase 8 — The NTT (Week 9)

Deliberately last. You can understand all of ML-DSA without it — the NTT is an
*optimization* of one operation. Learning it separately keeps it clean.

### Concepts

- The NTT as a DFT over a finite field instead of the complex numbers. Your
  signal-processing background transfers directly.
- The negacyclic NTT for `X^n + 1`, and why it needs a primitive 2n-th root of
  unity.
- **Cooley-Tukey** butterflies for the forward transform,
  **Gentleman-Sande** for the inverse. Why decimation-in-time forward plus
  decimation-in-frequency inverse lets you skip bit-reversal entirely.
- **Montgomery reduction** and **Barrett reduction**. Why you need them: q is
  23 bits, so coefficient products reach **46 bits**.
- Lazy reduction — keeping values in a wider range and reducing only when
  necessary.

### Build milestone

Replace your schoolbook `poly_mul` with an NTT-based one. Verify it produces
identical output on random inputs. Measure the speedup in Python.

Then: precompute the zeta table and check it against the one in the pq-crystals
reference. If they match, you have the root of unity and the ordering right.

### Resources

- **Longa & Naehrig**, "Speeding up the Number Theoretic Transform for Faster
  Ideal Lattice-Based Cryptography" (2016).
- The `ntt.c` file in the pq-crystals reference implementation, read alongside
  your own code.

---

## Phase 9 — Validation and reference reading (Week 10)

### The real milestone

**Make your Python implementation pass NIST's official ACVP test vectors for
ML-DSA.** Until KATs pass, you do not know your implementation is correct — you
only know it is self-consistent. Almost every bug that survives to this stage is
in encoding, packing, or domain separation.

This is the moment you can honestly say you understand the algorithm.

### Then read the C

Read **pq-crystals/dilithium** side by side with your own Python. Note every
place they differ, and work out why. Most differences are performance or
constant-time considerations — which is exactly the material you will need for
the hardware work.

Pay attention to:
- `poly.c`, `polyvec.c` — the data structures
- `ntt.c`, `reduce.c` — the arithmetic
- `sign.c` — the main flow, and how the rejection loop is structured
- `packing.c` — the encoding you probably got wrong first time

### Optional depth

- Constant-time implementation concerns: which branches depend on secrets, and
  what the reference does about it.
- Known side-channel attacks on Dilithium implementations.
- The security proof structure in the round-3 spec (skim; you need the shape,
  not the details).

---

## Milestone checklist

- [ ] Negacyclic polynomial ring in Python; `X^255 · X == -1` passes
- [ ] SHAKE XOF wrapper working
- [ ] Schnorr signatures implemented
- [ ] Private key recovered from two nonce-reusing Schnorr signatures
- [ ] Toy LWE implemented; Gaussian elimination observed to fail
- [ ] Proto-Dilithium: Schnorr over the ring, with rejection sampling
- [ ] Acceptance rate matches theory
- [ ] ML-DSA-44 KeyGen/Sign/Verify in Python, schoolbook multiplication
- [ ] 1000-message sign/verify round trip; bit-flip test fails correctly
- [ ] NTT implemented; matches schoolbook output
- [ ] Zeta table matches pq-crystals
- [ ] **ACVP known-answer test vectors pass**
- [ ] pq-crystals C reference read and diffed against own implementation

---

## Compressed path (6 weeks)

If time forces it, cut in this order:
1. Phase 6 (PQ motivation) — read on a train, half a day
2. Phase 4 depth — read only Peikert chapters 1-2, skip LLL entirely
3. Phase 1 formal ring theory — learn the negacyclic rule operationally

**Never cut Phase 3 (Schnorr) or Phase 9 (KAT validation).** The first is what
makes the algorithm comprehensible; the second is what proves you got it right.

---

## After this roadmap

Only then return to the hardware question. You will be in a position to answer
it properly, because you will know from your own profiling exactly where the
cycles go — rather than taking my word that Keccak dominates.

The framing to return to:

> ML-DSA has two kernels with opposite hardware appetites. NTT vectorizes well
> at 32-bit; Keccak does not vectorize on an embedded RVV unit. Given a fixed
> area budget, how should it be split between vector lanes and a dedicated
> Keccak unit?
