"""The explanation texts, keyed by topic and detail level (0 plain, 1
student, 2 expert).  HTML fragments; kept out of the widget code so they
can be edited without touching Qt."""

LEVEL_NAMES = ("Plain", "Student", "Expert")

STEP_TEXT = {
    "load": [
        """<p><b>Loading the block.</b> Before the stirring starts, the message chunk has to get into the
        cube. First you see the chunk by itself, sitting on the "bus" - the wires that carry it in. Then
        the cube as it was before (all dark the first time). Then, cycle by cycle, the bus delivers a
        few words and each one is merged into its place.</p>""",
        """<p><b>Loading (absorb) phase.</b> Frame 1: the padded block alone (the "seed" for this call).
        Frame 2: the state register before the absorb (the "initial value": all zero for the first
        block, the previous permutation's output afterwards). Then one frame per bus cycle:
        <code>state[word] ^= block[word]</code> for the words delivered in that cycle. Only the rate
        part (the first r bits) is ever written; the capacity stays as it was.</p>
        <p>The word size and the number of words per cycle are parameters - a 64-bit bus delivering one
        lane per cycle needs 17 cycles for SHA3-256's 1088-bit block; a 1088-bit bus loads it at once.</p>""",
        """<p><b>Loading, expert.</b> A hardware core's absorb is an XOR of the block into the rate part of
        the state register; with a narrow input bus this is serialised over ⌈r / bus⌉ cycles, which
        often dominates latency for short messages (17 cycles of I/O vs 24 cycles of permutation at one
        round per cycle). Padding is usually applied by the wrapper before the bus. Squeeze reads the
        rate words back over the same bus. Nothing in the loading phase depends on the data of other
        words, so the cycles can be pipelined with the previous permutation's output read-out.</p>""",
    ],
    "initial": [
        # plain
        """<p>This is the <b>starting cube</b>: 1600 tiny switches, each either off (dark) or on (bright).
        Your message has just been mixed into the front part of it. Everything else is still dark.</p>
        <p>What follows is a fixed recipe of <b>24 stirring rounds</b>. Each round has five moves. The
        recipe never changes; only the switches it operates on change - and that is enough to turn any
        message into an unpredictable pattern.</p>""",
        # student
        """<p>The state is a 5×5×64 array of bits <i>a[x][y][z]</i>: 25 <b>lanes</b> of 64 bits.
        Lane (x, y) is the 64-bit word at byte offset 8·(x+5y) of the 200-byte state, little-endian.</p>
        <p>Before the first permutation call the state is all zeros XOR the first padded message block
        (the first <i>r</i> bytes, the <b>rate</b>). The remaining <i>c</i> bytes (the <b>capacity</b>)
        start at zero and are never written by the message.</p>
        <p>Each round applies θ, ρ, π, χ, ι in that order. Use → to step through them.</p>""",
        # expert
        """<p>Keccak-f[1600] = Keccak-p[1600, 24]: b = 1600 = 25·2<sup>ℓ</sup> with ℓ = 6, n<sub>r</sub> = 12 + 2ℓ = 24.
        Bit string index i ↔ (x, y, z) with i = 64·(x + 5y) + z (FIPS 202 §3.1.2).</p>
        <p>The permutation is a bijection on {0,1}<sup>1600</sup>; θ, ρ, π, ι are GF(2)-linear (ι affine),
        χ is the only nonlinear map. Algebraic degree per round is 2, so after r rounds the degree is at
        most 2<sup>r</sup>; the inverse round has degree 3 (χ<sup>-1</sup>). 24 rounds leave a ~2× margin
        over the best published distinguishers (round-reduced results reach 8–9 rounds for the
        permutation, fewer for the hash).</p>""",
    ],
    "theta": [
        """<p><b>θ (theta) - mixing each column with its neighbours.</b></p>
        <p>Look at one vertical column of five switches. θ counts how many are on in the column just to
        the left and in the column to the right-and-one-step-back, and if that count is odd it flips
        the whole column.</p>
        <p>Every switch is influenced by ten neighbours at once. This is the move that spreads a change
        across the cube fastest - the animation shows the count sheets (parity) appearing below the
        cube, then the columns that flip.</p>""",
        """<p><b>θ: column parity diffusion</b></p>
        <pre>C[x]    = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4]     (parity of sheet x, per z)
D[x]    = C[x-1] ^ ROT(C[x+1], 1)                        (indices mod 5, rotation along z)
A'[x,y] = A[x,y] ^ D[x]</pre>
        <p>Every output bit is the XOR of 11 input bits: itself, the five bits of column (x−1, z) and
        the five bits of column (x+1, z−1). D[x] does not depend on y, so θ flips whole columns at
        once - watch for entire vertical columns lighting up in "changed" colour mode.</p>
        <p>The animation: first the <span style="color:#59d9f2">C sheet</span> (5×64 parities) forms under
        the cube, then the <span style="color:#a68cff">D sheet</span> (each cell XORs two C cells, one of
        them from the neighbouring slice), then the columns where D = 1 flip.</p>""",
        """<p><b>θ: expert notes</b></p>
        <p>Linear map with a very sparse matrix (weight 11 per row) but a dense inverse (θ<sup>-1</sup>
        has about half the state in each output bit), so backward diffusion is far stronger than
        forward - relevant when reasoning about inside-out attacks.</p>
        <p>The <b>column-parity kernel</b> (states where every column has even parity, C ≡ 0) is fixed by
        θ; low-weight differential trails try to stay in the kernel, which is why ρ and π are chosen to
        push a kernel state out of the kernel within one round.</p>
        <p><b>Hardware:</b> C needs 5·64 five-input parities = 1280 XOR2 (3 levels balanced); D adds
        320 XOR2 (1 level; the ROT is wiring); the final add is 1600 XOR2. ≈ 3200 XOR2 per round, ~5 XOR
        levels of critical path, no state kept. It dominates the round's XOR count and its depth. In
        software it is 5 lane XORs for C, 5 for D (plus 5 rotates) and 25 for the update.</p>""",
    ],
    "rho": [
        """<p><b>ρ (rho) - sliding each lane.</b></p>
        <p>Each of the 25 long lanes slides along its length by its own fixed amount, wrapping around
        at the end like a conveyor belt. Lane (0,0) does not move; the others move by 1, 3, 6, 10, ...
        up to 62 places.</p>
        <p>By itself this changes nothing about which switches are on - it only moves them. But it
        means the next θ will mix switches that used to be far apart.</p>""",
        """<p><b>ρ: lane rotations</b></p>
        <pre>A'[x,y] = ROT(A[x,y], r[x,y])       (bit z moves to z + r[x,y] mod 64)</pre>
        <p>The offsets r[x,y] are the triangular numbers (t+1)(t+2)/2 mod 64 for t = 0…23, assigned by
        walking (x, y) → (y, 2x+3y) starting from (1, 0); (0, 0) gets offset 0. The lane table view
        lists r for each lane. All 25 offsets are distinct.</p>
        <p>ρ is the only step that moves bits <i>between slices</i> by more than one position (θ only
        reaches the previous slice). Without it, information would crawl along z one slice per round.</p>
        <p>In the animation each lane visibly slides along z; cells that fall off the far end reappear
        at the near end.</p>""",
        """<p><b>ρ: expert notes</b></p>
        <p>Pure wiring in hardware: 0 gates, 0 depth. In 64-bit software it is 24 rotate instructions
        (often folded into the θ/χ data flow with "lane complementing" tricks). On 32-bit targets use the
        <b>bit-interleaved</b> layout (see that view): ROL64 by r becomes two ROL32 by ⌊r/2⌋, ⌈r/2⌉ with a
        word swap when r is odd, so no 64-bit shifter is ever needed.</p>
        <p>The offsets are triangular numbers because the designers wanted the set of offsets to be
        well spread modulo 64 (the differences between offsets of lanes that interact in θ and χ should
        avoid small structure); the (x,y) ↦ (y, 2x+3y) orbit has order 24 and visits every lane except
        (0,0), which is why one orbit suffices to assign 24 distinct offsets.</p>""",
    ],
    "pi": [
        """<p><b>π (pi) - shuffling the lanes.</b></p>
        <p>The 25 lanes swap places according to a fixed pattern, like a dance where everybody moves to a
        new square. The lane in the corner (0,0) stays put.</p>
        <p>This makes sure that the "left/right neighbour" relationships used by the other moves keep
        changing, so no group of switches can stay isolated from the rest.</p>""",
        """<p><b>π: lane transposition</b></p>
        <pre>A'[y, 2x+3y] = A[x, y]        equivalently   A'[x,y] = A[(x+3y) mod 5, x]</pre>
        <p>A fixed permutation of the 25 lane positions, applied identically to every slice. The
        coloured arrows on the front and back faces show where each lane goes. It is an invertible
        linear map on (x, y) over Z<sub>5</sub> with matrix [[0,1],[2,3]].</p>
        <p>Why: χ works along rows (x) and θ along columns (y). π rotates the grid so that bits that were
        in the same row end up in different rows and columns next round; together with ρ this destroys
        any alignment of bit patterns.</p>""",
        """<p><b>π: expert notes</b></p>
        <p>Wiring only (0 gates). The matrix M = [[0,1],[2,3]] over Z<sub>5</sub> has det = −2 = 3 ≠ 0 and
        order 24 acting on Z<sub>5</sub><sup>2</sup>∖{0} as a single cycle - the same orbit used to
        assign ρ offsets, so the step sequence ρ∘π can be reasoned about along one orbit.</p>
        <p>π maps the row-alignment of χ into the column-alignment of θ: a low-weight state confined
        to a few rows is spread over many columns, defeating attempts to stay in θ's column-parity
        kernel for two consecutive rounds. In software π is register renaming (free in an unrolled
        loop); in the batched/SIMD layout it is likewise just a rename of vector registers.</p>""",
    ],
    "chi": [
        """<p><b>χ (chi) - the only genuinely scrambling move.</b></p>
        <p>Look along a row of five switches. Each switch looks at its two right-hand neighbours: if
        the first is off <i>and</i> the second is on, it flips. Everything else in the recipe is
        "just" moving or adding; this is the one move that behaves like a lock rather than a shuffle.</p>
        <p>Without it, you could undo the whole stirring with simple algebra.</p>""",
        """<p><b>χ: the nonlinear row S-box</b></p>
        <pre>A'[x,y] = A[x,y] ^ (¬A[x+1,y] &amp; A[x+2,y])      (x indices mod 5)</pre>
        <p>Acts independently on each 5-bit <b>row</b> (fixed y, z): 320 parallel 5-bit S-boxes. The AND
        makes it nonlinear (degree 2); the map on a row is invertible, so χ is a permutation. The
        animation pulses the cells that flip and draws the two feeding neighbours of each.</p>
        <p>Every other step is linear over GF(2); a hash made only of linear steps could be inverted by
        solving linear equations. χ is what gives Keccak its cryptographic strength; θ, ρ, π exist to
        spread χ's local nonlinearity across the whole state.</p>""",
        """<p><b>χ: expert notes</b></p>
        <p>Algebraic degree 2 forward, 3 for χ<sup>-1</sup> (the inverse of a 5-bit χ row is a
        polynomial of degree 3; for width 5 it needs 3 iterations of the "χ trick"). Row S-box
        properties: differential uniformity 8/32 (max DP = 2<sup>-2</sup>), best linear correlation
        2<sup>-1</sup>; the S-box is "shift-invariant" - one Boolean function replicated 5 times, which is
        what allows the bit-sliced / lane-wise implementation.</p>
        <p><b>Hardware:</b> per bit one AND (with inverted input - a single NAND/ANDN cell) and one XOR:
        1600 AND2 + 1600 XOR2 per round, 2 gate levels. The 1600 ANDs are the only nonlinear gates and
        set the cost of masked (side-channel-protected) implementations, where each AND becomes a
        gadget with fresh randomness. Software: 5 ANDN + 5 XOR per plane, 25 each per round.</p>""",
    ],
    "iota": [
        """<p><b>ι (iota) - stamping the round number.</b></p>
        <p>A tiny move: a few switches in one corner lane are flipped according to a fixed table, a
        different pattern each round. It looks harmless, but without it every round would be
        identical and the cube would have "blind spots" - for example an all-dark cube would stay dark
        forever.</p>""",
        """<p><b>ι: round constant</b></p>
        <pre>A'[0,0] = A[0,0] ^ RC[i<sub>r</sub>]</pre>
        <p>Only lane (0,0) changes, in at most 7 bit positions: 0, 1, 3, 7, 15, 31, 63 (2<sup>j</sup>−1).
        RC[i<sub>r</sub>] comes from a small LFSR, so the 24 constants look random but are cheap to
        generate. The lane table view prints the constant of the current round.</p>
        <p>Purpose: break symmetry. θ, ρ, π, χ all map the all-zero state to itself and commute with
        translation along z; ι destroys both properties so that no simple structure survives 24
        rounds.</p>""",
        """<p><b>ι: expert notes</b></p>
        <p>RC[i<sub>r</sub>][2<sup>j</sup>−1] = rc(j + 7·i<sub>r</sub>) where rc is the output of the
        8-bit LFSR with polynomial x<sup>8</sup>+x<sup>6</sup>+x<sup>5</sup>+x<sup>4</sup>+1 (FIPS 202
        Alg. 5/6). Keccak-p[1600, n<sub>r</sub>] with n<sub>r</sub> &lt; 24 uses the <i>last</i>
        n<sub>r</sub> constants (rounds 24−n<sub>r</sub> … 23) - this tool follows that convention
        (toggle in the parameter panel).</p>
        <p>Without ι the round function is invariant under z-translation of the whole state (slide
        properties) and under some symmetric configurations; ι also removes the fixed point 0.
        <b>Hardware:</b> ≤ 7 XOR2 per round, usually merged into χ's XOR for those bits; a 24-entry ROM
        or the LFSR itself for the constants. In an unrolled core the constants are literals.</p>""",
    ],
}

MODE_TEXT = {
    "cube": [
        """<p><b>The cube.</b> 5 wide (x), 5 tall (y), 64 deep (z). Bright cubes are switches that are on.
        Drag to look around, scroll to zoom, click a cube to see who influences it next. Hover the
        coloured words at the bottom to see what a <i>row</i>, <i>column</i>, <i>lane</i>, <i>slice</i>,
        <i>plane</i> and <i>sheet</i> are.</p>""",
        """<p><b>3D cube.</b> Cell (x, y, z) sits at those coordinates; lanes run along z. Colour modes:
        raw 0/1; <i>changed</i> (red = flipped to 1, blue = flipped to 0 since the previous step);
        <i>avalanche</i> (magenta = differs from the run with one input bit flipped). The substructure
        chips highlight a row (5 bits along x), column (5 along y), lane (64 along z), slice (25 bits
        at one z), plane (5 lanes, fixed y) or sheet (5 lanes, fixed x).</p>""",
        """<p><b>3D cube.</b> One instanced draw call for the 1600 cells (position, colour, scale per
        instance), a second for lines; the per-frame instance buffer is computed in NumPy from the two
        snapshots being interpolated. The camera presets are orthographic (slice-on, lane-on, top) so that
        aligned structures overlap exactly. Feed lines for a selected cell follow the exact
        <code>sources_of()</code> relation of the next step in the trace.</p>""",
    ],
    "words": [
        """<p><b>Words.</b> The same 1600 switches grouped into words of a chosen size, written as
        numbers. Words highlighted in orange were touched by the last move; cyan ones are arriving on
        the bus.</p>""",
        """<p><b>Words.</b> Word w covers bits w·W … w·W+W−1 of the state string (bit i = bit z of lane
        (x,y) with i = 64·(x+5y)+z), so words never straddle lanes for W ≤ 64. The first r/W words are
        the rate: the only ones absorb writes and squeeze reads. Choose W = 32 to see the two halves an
        implementation on a 32-bit CPU handles, W = 8 for the byte view.</p>""",
        """<p><b>Words.</b> Useful for checking a datapath that processes the state W bits at a time:
        compare the highlighted changed words per step with your simulation. Note ρ changes every
        word of a rotated lane while θ's D[x] flips whole columns and thus touches every word of a
        sheet. Word bands can be shaded on the 3D cube via the display menu.</p>""",
    ],
    "slices": [
        """<p><b>Slice stack.</b> The cube cut into its 64 slices and laid out flat, so you can see every
        switch at once. θ and χ never reach outside a slice; ρ is the move that shuffles between slices.</p>""",
        """<p><b>Slice stack.</b> Each 5×5 square is slice z (x to the right, y up). θ's effect appears as
        whole columns flipping inside a slice, χ's as row-local changes; after ρ the picture shifts
        between slices; after π each slice is permuted identically. Click a cell to select it and see
        its feeding cells outlined in cyan.</p>""",
        """<p><b>Slice stack.</b> Useful for trail analysis: an active slice pattern is what the
        column-parity kernel and χ's row propagation act on; count active rows/columns per slice to
        reason about weight. The C/D intermediates of θ are per-(x, z), i.e. per column of this
        layout.</p>""",
    ],
    "lanes": [
        """<p><b>Lane table.</b> The same cube written as 25 long numbers (hexadecimal). This is how a
        programmer sees Keccak.</p>""",
        """<p><b>Lane table.</b> A[x,y] as 64-bit words, x across, y down, so lane index x + 5y counts left
        to right, top to bottom. The XOR with the previous step shows exactly which bits each step
        touched; θ's C[x] and D[x] and ι's RC are printed for the current step. "copy table as text"
        puts a debug-friendly dump on the clipboard.</p>""",
        """<p><b>Lane table.</b> Matches the intermediate-value format of the Keccak reference
        (KeccakF-1600-IntermediateValues): compare after each step against a hardware simulation. Lane
        words are little-endian in the byte view, so lane (x,y) printed here equals the uint64 read at
        byte offset 8·(x+5y).</p>""",
    ],
    "bytes": [
        """<p><b>State bytes.</b> The cube as 200 bytes. The green part is where your message goes in
        and where the answer comes out; the purple part is the sponge's hidden reservoir.</p>""",
        """<p><b>State bytes.</b> The 200-byte buffer, one lane per row. The yellow line is the
        rate/capacity boundary of the chosen variant: absorb XORs a block into bytes 0…r−1 and squeeze
        reads bytes 0…r−1; bytes r…199 (capacity) are only ever changed by the permutation. Larger
        capacity = more security, smaller rate = slower hashing.</p>""",
        """<p><b>State bytes.</b> Bit i of the state string is bit (i mod 8) of byte ⌊i/8⌋, LSB first;
        the padded block XORs in with the domain byte at message offset and 0x80 at byte r−1. Variant
        rates: SHA3-224 144, SHA3-256 136, SHA3-384 104, SHA3-512 72, SHAKE128 168, SHAKE256 136. The
        capacity c gives 2<sup>c/2</sup> generic (pre-image/collision) security for the sponge.</p>""",
    ],
    "heatmap": [
        """<p><b>Diffusion heatmap.</b> Flip one switch at the start and ask: how long until each other
        switch notices? Bright = very soon, dark = later. After about three rounds every switch has
        noticed.</p>""",
        """<p><b>Diffusion heatmap.</b> Two runs, differing in one input bit; each cell shows the step at
        which the two runs first differ there. θ reaches 11 cells in one step, χ doubles the spread
        along rows, ρ/π scatter it; full diffusion (all 1600 bits) typically by round 3–4. Try
        disabling θ in the parameters to see diffusion crawl.</p>""",
        """<p><b>Diffusion heatmap.</b> This is a single-sample, data-dependent measure (χ propagates a
        difference through an AND only when the neighbouring bit is set), i.e. an empirical bound on
        the full-diffusion round count. The linear part alone (θ,ρ,π) has a deterministic dependency
        cone; averaged over inputs the expected first-affected round is what matters for the
        "propagation" criterion used in the Keccak design rationale.</p>""",
    ],
    "avalanche": [
        """<p><b>Avalanche.</b> Change one switch at the start and compare the two cubes as the stirring
        proceeds. The graph counts how many switches differ: a good mixer quickly reaches "about half of
        them" and stays there - flip one bit in, and half the bits change out.</p>""",
        """<p><b>Avalanche.</b> Hamming distance between the two runs after each round: from 1 to ≈ 800
        (half of 1600) within 3 rounds, then hovering around 800 with a spread of a few tens - the
        strict avalanche criterion. Untick θ in the parameters: the curve collapses because χ alone
        spreads a difference only within its row and ρ/π only move it.</p>""",
        """<p><b>Avalanche.</b> The expected Hamming distance for a random permutation is 800 with
        standard deviation 20 (binomial 1600 × ½). Per-step plotting shows that θ contributes the jump
        and χ the "randomisation" of the pattern. Reduced-round experiments: with 1–2 rounds the
        distance is far below 800, which is what round-reduced distinguishers exploit.</p>""",
    ],
    "interleaved": [
        """<p><b>Bit-interleaved.</b> A trick for small computers that can only handle 32-bit numbers:
        split every 64-bit lane into two halves - not left/right, but even/odd positions. Then the
        "slide" move becomes two smaller slides.</p>""",
        """<p><b>Bit-interleaved.</b> E holds bits 0,2,4,… of the lane, O holds bits 1,3,5,… . XOR, AND and
        NOT are bitwise so they work unchanged on E and O. A 64-bit rotate by r: if r is even both words
        rotate by r/2; if r is odd the words swap roles, E' = ROL32(O, (r+1)/2) and O' = ROL32(E, (r−1)/2).
        The worked example at the bottom checks this on the selected lane.</p>""",
        """<p><b>Bit-interleaved.</b> Standard on Cortex-M / 32-bit RISC-V (XKCP "Inplace32BI"): the
        interleave/de-interleave costs a few bit-permute passes at absorb and squeeze only; inside the
        permutation every ROL64 becomes ROL32 with zero extra instructions, and θ's ROT(C, 1) is a swap
        plus one ROL32 by 1. Register pressure: 50 words of state, so the state lives in memory and
        the round is processed plane by plane.</p>""",
    ],
    "batched": [
        """<p><b>Batched lanes.</b> Hashing several messages at once: put the same piece of each cube
        side by side in one wide register. Every move then happens to all of them together, and none
        of the moves ever needs to look sideways within the register.</p>""",
        """<p><b>Batched lanes.</b> N independent states; vector register R[x,y] holds lane (x,y) of
        instance 0…N−1. θ's parity C[x] = ⊕<sub>y</sub> R[x,y] is an element-wise XOR of five registers:
        element i of C[x] depends only on element i of the inputs. ρ is a per-element rotate, π renames
        registers, χ is register-wide logic, ι broadcasts a constant. No lane crossing anywhere.</p>""",
        """<p><b>Batched lanes.</b> This is the AVX2/AVX-512/NEON ×N layout (e.g. Keccak×4 with 64-bit
        elements in 256-bit registers). Cost per round per N instances = the scalar instruction count,
        so throughput scales with the vector width until load/store bandwidth limits it. Contrast with
        the NTT in ML-DSA whose butterflies need cross-lane permutes at log<sub>2</sub>(VL) of the
        stages. ML-DSA signing calls SHAKE many times with independent inputs, which is exactly this
        pattern.</p>""",
    ],
    "sponge": [
        """<p><b>The sponge.</b> A hash function built from the stirring cube: pour the message in one
        cupful at a time, stir thoroughly after each cup, then wring the sponge to get the answer out.
        Cups that are not full are topped up with a fixed pattern (the padding).</p>""",
        """<p><b>The sponge construction.</b> State = rate r + capacity c = 1600 bits. <b>Padding</b>:
        append the domain suffix (01 for SHA-3, 1111 for SHAKE, nothing for original Keccak), then
        pad10*1 to a multiple of r; the domain byte merges suffix and first pad bit (0x06 / 0x1F /
        0x01), the last pad bit is 0x80 in the block's last byte. <b>Absorb</b>: for each block, XOR it
        into the first r bits and apply f. <b>Squeeze</b>: read r bits of output; if more is needed,
        apply f and read again. The permutation counter shows the cost.</p>""",
        """<p><b>Sponge, expert.</b> Indifferentiable from a random oracle up to ~2<sup>c/2</sup> queries
        (Bertoni et al. 2008); pre-image, second pre-image and collision resistance are min(2<sup>c/2</sup>,
        2<sup>n</sup>, …) accordingly - SHA3-256 has c = 512 so 256-bit collision resistance, SHAKE128
        has c = 256. Padding is suffix-free and the domain bits keep SHA3-256 and SHAKE256 (same r)
        unrelated. The 0x80 / domain-byte merge happens when the message fills the block to its last
        byte. Absorb and squeeze are the only places the outside world touches the state, and only the
        rate part.</p>""",
    ],
}

GLOSSARY = """
<p style="color:#9aa"><b>Vocabulary</b> (FIPS 202 fig. 2): <b>row</b> = 5 bits along x · <b>column</b> = 5
bits along y · <b>lane</b> = 64 bits along z · <b>slice</b> = 25 bits at one z · <b>plane</b> = 5 lanes
with equal y · <b>sheet</b> = 5 lanes with equal x.</p>
"""
