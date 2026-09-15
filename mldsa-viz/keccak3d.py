#!/usr/bin/env python3
"""
keccak3d.py -- interactive 3D visualization of the SHA-3 (Keccak-f[1600]) permutation.

The Keccak state is a 5 x 5 x 64 array of bits: 25 lanes of 64 bits each.
This script runs a real SHA3-256 absorb on your message, then steps through all
24 rounds of the permutation, one step mapping at a time, drawing the live bit
lattice in 3D.

  theta  - column parity diffusion   (each bit XORs two neighbouring columns)
  rho    - per-lane rotation along z (inter-slice dispersion)
  pi     - lane permutation within slices
  chi    - the only nonlinear step, acting along rows (x)
  iota   - round constant XORed into lane (0,0), breaking symmetry

Requires: numpy, matplotlib.
    pip install numpy matplotlib

Usage:
    python keccak3d.py                       # visualize SHA3-256("")
    python keccak3d.py -m "hello world"      # visualize a message
    python keccak3d.py --selftest            # verify against hashlib, no GUI
    python keccak3d.py --avalanche           # start in diffusion-difference mode

Keys (in the window):
    right / space   next step mapping
    left            previous step mapping
    up / down       jump forward / back one full round
    p               toggle autoplay
    a               toggle avalanche (diff vs. 1-bit-flipped message)
    g               toggle the geometry guide (column / lane / slice / row)
    0               jump to the initial absorbed state
    q               quit
"""

import argparse
import hashlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Keccak-f[1600] core
# ---------------------------------------------------------------------------

MASK = (1 << 64) - 1

# Rotation offsets r[x][y] for the rho step.
ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

# Round constants for iota.
RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

STEPS = ("theta", "rho", "pi", "chi", "iota")


def rol(value, n):
    """Rotate a 64-bit lane left by n bits."""
    n %= 64
    if n == 0:
        return value & MASK
    return ((value << n) | (value >> (64 - n))) & MASK


def new_state():
    return [[0] * 5 for _ in range(5)]  # A[x][y]


def copy_state(a):
    return [row[:] for row in a]


def keccak_f_snapshots(state, n_rounds=24):
    """Run Keccak-f[1600], returning (final_state, snapshots).

    snapshots is a list of (round_index, step_name, state) with the initial
    state recorded as (-1, 'absorb', ...). 24 rounds x 5 steps + 1 = 121 entries.
    """
    A = copy_state(state)
    snaps = [(-1, "absorb", copy_state(A))]

    for rnd in range(n_rounds):
        # -- theta: column parity ------------------------------------------
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ rol(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                A[x][y] ^= D[x]
        snaps.append((rnd, "theta", copy_state(A)))

        # -- rho: lane rotation along z ------------------------------------
        for x in range(5):
            for y in range(5):
                A[x][y] = rol(A[x][y], ROT[x][y])
        snaps.append((rnd, "rho", copy_state(A)))

        # -- pi: lane permutation ------------------------------------------
        B = new_state()
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = A[x][y]
        A = B
        snaps.append((rnd, "pi", copy_state(A)))

        # -- chi: nonlinear step along rows --------------------------------
        B = copy_state(A)
        for x in range(5):
            for y in range(5):
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y] & MASK) & B[(x + 2) % 5][y])
        snaps.append((rnd, "chi", copy_state(A)))

        # -- iota: round constant ------------------------------------------
        A[0][0] ^= RC[rnd]
        snaps.append((rnd, "iota", copy_state(A)))

    return A, snaps


def keccak_f(state):
    return keccak_f_snapshots(state)[0]


# ---------------------------------------------------------------------------
# SHA3-256 sponge (rate = 1088 bits = 136 bytes, capacity = 512 bits)
# ---------------------------------------------------------------------------

RATE_BYTES = 136


def pad10star1(data):
    """SHA-3 domain separation (0x06) plus pad10*1 to a multiple of the rate."""
    padded = bytearray(data)
    padded.append(0x06)
    while len(padded) % RATE_BYTES != 0:
        padded.append(0x00)
    padded[-1] |= 0x80
    return bytes(padded)


def absorb_block(state, block):
    """XOR one rate-sized block into the state (lane index i = x + 5y)."""
    A = copy_state(state)
    for i in range(RATE_BYTES // 8):
        lane = int.from_bytes(block[i * 8:(i + 1) * 8], "little")
        A[i % 5][i // 5] ^= lane
    return A


def sha3_256(data):
    """Full SHA3-256, used only to prove the core above is correct."""
    A = new_state()
    for off in range(0, len(pad10star1(data)), RATE_BYTES):
        A = keccak_f(absorb_block(A, pad10star1(data)[off:off + RATE_BYTES]))
    out = b""
    i = 0
    while len(out) < 32:
        out += A[i % 5][i // 5].to_bytes(8, "little")
        i += 1
    return out[:32]


def first_block_state(message):
    """State right after absorbing the first block -- the input to round 0."""
    padded = pad10star1(message)
    return absorb_block(new_state(), padded[:RATE_BYTES]), len(padded) // RATE_BYTES


def selftest():
    vectors = [b"", b"abc", b"hello world", bytes(range(200)), b"x" * 136]
    ok = True
    for v in vectors:
        mine = sha3_256(v).hex()
        ref = hashlib.sha3_256(v).hexdigest()
        status = "ok " if mine == ref else "FAIL"
        if mine != ref:
            ok = False
        label = repr(v if len(v) <= 16 else v[:13] + b"...")
        print(f"[{status}] {label:<24} {mine}")
    print("\nall vectors match hashlib" if ok else "\nMISMATCH -- core is wrong")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Bit extraction
# ---------------------------------------------------------------------------

def state_to_bits(state):
    """Return a (5, 5, 64) uint8 array indexed [x][y][z]."""
    lanes = np.array([[state[x][y] for y in range(5)] for x in range(5)],
                     dtype=object)
    bits = np.zeros((5, 5, 64), dtype=np.uint8)
    for x in range(5):
        for y in range(5):
            lane = int(lanes[x][y])
            for z in range(64):
                bits[x, y, z] = (lane >> z) & 1
    return bits


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

STEP_TEXT = {
    "absorb": "message block XORed into the state (rate lanes only)",
    "theta": "theta -- each bit XORs the parity of two neighbouring columns",
    "rho":   "rho -- every lane rotated along z by its own fixed offset",
    "pi":    "pi -- lanes permuted within each slice: (x,y) -> (y, 2x+3y)",
    "chi":   "chi -- the nonlinear step, applied independently to each row",
    "iota":  "iota -- round constant XORed into lane (0,0) only",
}

# Indices used by the geometry guide, purely illustrative.
GX, GY, GZ = 2, 2, 32


class KeccakViewer:
    def __init__(self, snaps, snaps_alt, message, n_blocks, z_squash=0.35):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.snaps = snaps
        self.snaps_alt = snaps_alt
        self.message = message
        self.n_blocks = n_blocks
        self.z_squash = z_squash

        print("precomputing bit lattices ...", flush=True)
        self.bits = [state_to_bits(s) for _, _, s in snaps]
        self.bits_alt = [state_to_bits(s) for _, _, s in snaps_alt]

        self.idx = 0
        self.diff_mode = False
        self.show_guide = True
        self.playing = False

        xi, yi, zi = np.indices((5, 5, 64))
        self.px = zi.ravel() * z_squash
        self.py = xi.ravel().astype(float)
        self.pz = yi.ravel().astype(float)

        self.fig = plt.figure(figsize=(14, 6.5))
        self.fig.canvas.manager.set_window_title("SHA-3 / Keccak-f[1600]")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=220)
        self.timer.add_callback(self._tick)

        self.elev, self.azim = 18, -62
        self.draw()

    # -- helpers -----------------------------------------------------------

    def _tick(self):
        if self.playing:
            if self.idx < len(self.snaps) - 1:
                self.idx += 1
                self.draw()
            else:
                self.playing = False
                self.timer.stop()
                self.draw()

    def guide_segments(self, step):
        """Line segments illustrating the structure the current step acts on."""
        s, segs = self.z_squash, []
        if step == "theta":
            for x, col in ((GX, "target"), ((GX - 1) % 5, "src"), ((GX + 1) % 5, "src")):
                segs.append(([GZ * s, GZ * s], [x, x], [0, 4], col))
        elif step in ("rho", "iota"):
            x, y = (GX, GY) if step == "rho" else (0, 0)
            segs.append(([0, 63 * s], [x, x], [y, y], "target"))
        elif step == "pi":
            sq = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
            for (a, b), (c, d) in zip(sq, sq[1:]):
                segs.append(([GZ * s, GZ * s], [a, c], [b, d], "target"))
        elif step == "chi":
            segs.append(([GZ * s, GZ * s], [0, 4], [GY, GY], "target"))
        return segs

    # -- drawing -----------------------------------------------------------

    def draw(self):
        ax = self.ax
        self.elev, self.azim = ax.elev, ax.azim
        ax.clear()

        rnd, step, _ = self.snaps[self.idx]
        cur = self.bits[self.idx]

        if self.diff_mode:
            field = (cur ^ self.bits_alt[self.idx]).ravel()
            title_mode = "avalanche: bits differing from 1-bit-flipped message"
        else:
            field = cur.ravel()
            title_mode = "state bits (1 = set)"

        on = field == 1
        off = ~on

        # unset lattice sites, faint, so the 5 x 5 x 64 volume stays legible
        ax.scatter(self.px[off], self.py[off], self.pz[off],
                   s=1.5, c="#8899aa", alpha=0.05, depthshade=False,
                   linewidths=0, marker=".")

        if self.diff_mode:
            colors = "#ff3b30"
        else:
            cmap = self.plt.get_cmap("viridis")
            colors = cmap(self.pz[on] / 4.0)

        ax.scatter(self.px[on], self.py[on], self.pz[on],
                   s=13, c=colors, alpha=0.9, depthshade=False,
                   linewidths=0, marker="o")

        if self.show_guide:
            for xs, ys, zs, kind in self.guide_segments(step):
                ax.plot(xs, ys, zs,
                        color="#ff9500" if kind == "target" else "#00b0ff",
                        linewidth=2.4 if kind == "target" else 1.6,
                        alpha=0.95, zorder=10)

        # bounding box of the state volume
        zmax = 63 * self.z_squash
        for a in (0, 4):
            for b in (0, 4):
                ax.plot([0, zmax], [a, a], [b, b], color="#445566", lw=0.7, alpha=0.5)
        for zz in (0, zmax):
            ax.plot([zz, zz, zz, zz, zz], [0, 4, 4, 0, 0], [0, 0, 4, 4, 0],
                    color="#445566", lw=0.7, alpha=0.5)

        ax.set_box_aspect((63 * self.z_squash, 4.6, 4.6))
        ax.set_xlabel("z  (lane bit, 0-63)", labelpad=12)
        ax.set_ylabel("x", labelpad=2)
        ax.set_zlabel("y", labelpad=2)
        ax.set_yticks(range(5))
        ax.set_zticks(range(5))
        ax.set_xticks([k * self.z_squash for k in (0, 16, 32, 48, 63)])
        ax.set_xticklabels(["0", "16", "32", "48", "63"])
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.grid(False)
        try:
            ax.set_facecolor("white")
            for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
                pane.pane.fill = False
                pane.pane.set_edgecolor("#dddddd")
        except Exception:
            pass

        weight = int(field.sum())
        pct = 100.0 * weight / 1600.0
        where = "absorbed block" if rnd < 0 else f"round {rnd:2d}/23  ->  {step}"
        msg = self.message if len(self.message) <= 28 else self.message[:25] + b"..."

        head = f"SHA3-256({msg!r})   {self.n_blocks} block(s)   |   step {self.idx}/{len(self.snaps) - 1}   |   {where}"
        sub = f"{STEP_TEXT[step]}\n{title_mode}: {weight}/1600 ({pct:.1f}%)"
        ax.set_title(head + "\n" + sub, fontsize=10, loc="left", pad=16, family="monospace")

        self.fig.text(0.985, 0.02,
                      "arrows step/round   space next   p play   a avalanche   g guide   0 reset   q quit",
                      ha="right", va="bottom", fontsize=8, color="#667788",
                      family="monospace")
        self.fig.canvas.draw_idle()

    # -- input -------------------------------------------------------------

    def on_key(self, event):
        k = event.key
        last = len(self.snaps) - 1
        if k in ("right", " "):
            self.idx = min(self.idx + 1, last)
        elif k == "left":
            self.idx = max(self.idx - 1, 0)
        elif k == "up":
            self.idx = min(self.idx + 5, last)
        elif k == "down":
            self.idx = max(self.idx - 5, 0)
        elif k == "0":
            self.idx = 0
        elif k == "a":
            self.diff_mode = not self.diff_mode
        elif k == "g":
            self.show_guide = not self.show_guide
        elif k == "p":
            self.playing = not self.playing
            self.timer.start() if self.playing else self.timer.stop()
        elif k == "q":
            self.plt.close(self.fig)
            return
        else:
            return
        self.draw()

    def show(self):
        self.fig.text(0.015, 0.02,
                      "orange = structure the current step acts on;  blue = its sources",
                      ha="left", va="bottom", fontsize=8, color="#667788",
                      family="monospace")
        self.plt.show()


# ---------------------------------------------------------------------------

def flip_first_bit(data):
    b = bytearray(data) if data else bytearray(b"\x00")
    b[0] ^= 0x01
    return bytes(b)


def main():
    ap = argparse.ArgumentParser(description="3D visualization of SHA-3 / Keccak-f[1600].")
    ap.add_argument("-m", "--message", default="", help="message to hash (default: empty)")
    ap.add_argument("--selftest", action="store_true", help="verify core against hashlib and exit")
    ap.add_argument("--avalanche", action="store_true", help="start in difference mode")
    ap.add_argument("--squash", type=float, default=0.35,
                    help="compress the 64-bit z axis for readability (1.0 = true scale)")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    msg = args.message.encode()
    digest = sha3_256(msg)
    assert digest == hashlib.sha3_256(msg).digest(), "core mismatch"
    print(f"SHA3-256({msg!r}) = {digest.hex()}")

    state, n_blocks = first_block_state(msg)
    _, snaps = keccak_f_snapshots(state)

    alt_state, _ = first_block_state(flip_first_bit(msg))
    _, snaps_alt = keccak_f_snapshots(alt_state)

    if n_blocks > 1:
        print(f"note: message spans {n_blocks} blocks; visualizing the first permutation call.")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is required:  pip install numpy matplotlib", file=sys.stderr)
        return 1

    v = KeccakViewer(snaps, snaps_alt, msg, n_blocks, z_squash=args.squash)
    v.diff_mode = args.avalanche
    v.draw()
    v.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
