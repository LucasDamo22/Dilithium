#!/usr/bin/env python3
"""
ntt3d.py -- the Dilithium / ML-DSA NTT butterfly network, laid out in
VECTOR REGISTER SPACE rather than the usual flat index-vs-stage plot.

The point of this layout: on a vector machine the 256 coefficients live in
ceil(256/VL) registers of VL elements each. A butterfly pairs index j with
index j+len. Whether that pair sits in the SAME LANE of two different
registers, or in DIFFERENT LANES, decides whether the stage is a free
element-wise vector op or needs a slide / gather / vrgather permute.

    x axis = lane      (j mod VL)
    y axis = register  (j div VL)
    z axis = stage     (0 .. 7, len = 128, 64, 32, 16, 8, 4, 2, 1)

A butterfly is drawn as a segment joining its two partners in the plane of
its stage. Teal = same lane (pure element-wise). Coral = crosses lanes
(needs a permute). The colour flips exactly when len < VL, which is the
cost cliff every RVV NTT implementation runs into.

The NTT itself is the real one: n = 256, q = 8380417, zeta = 1753,
negacyclic, verified against schoolbook convolution.

Requires: numpy, matplotlib  ->  pip install numpy matplotlib

Usage:
    python ntt3d.py                  # VL = 8  (VLEN=256, SEW=32)
    python ntt3d.py --vl 4
    python ntt3d.py --selftest       # verify the transform, no GUI
    python ntt3d.py --table          # print the stage cost table for many VL

Keys:  1 2 3 4 5 6 -> VL = 2 4 8 16 32 64     s -> toggle stage labels
       up / down   -> isolate one stage / show all      q -> quit
"""

import argparse
import sys

import numpy as np

Q = 8380417
N = 256
ZETA = 1753


def brv8(i):
    return int(f"{i:08b}"[::-1], 2)


ZETAS = [pow(ZETA, brv8(i), Q) for i in range(N)]


def ntt(a):
    """In-place Cooley-Tukey negacyclic NTT, reference Dilithium structure."""
    a = list(a)
    k, length = 0, 128
    while length >= 1:
        start = 0
        while start < N:
            k += 1
            z = ZETAS[k]
            for j in range(start, start + length):
                t = z * a[j + length] % Q
                a[j + length] = (a[j] - t) % Q
                a[j] = (a[j] + t) % Q
            start += 2 * length
        length >>= 1
    return a


def invntt(a):
    a = list(a)
    k, length = N, 1
    while length <= 128:
        start = 0
        while start < N:
            k -= 1
            z = (-ZETAS[k]) % Q
            for j in range(start, start + length):
                t = a[j]
                a[j] = (t + a[j + length]) % Q
                a[j + length] = z * (t - a[j + length]) % Q
            start += 2 * length
        length <<= 1
    ninv = pow(N, Q - 2, Q)
    return [x * ninv % Q for x in a]


def schoolbook(a, b):
    """Negacyclic convolution in Z_q[x]/(x^256 + 1), the slow way."""
    r = [0] * N
    for i in range(N):
        if a[i] == 0:
            continue
        for j in range(N):
            k = i + j
            if k < N:
                r[k] = (r[k] + a[i] * b[j]) % Q
            else:
                r[k - N] = (r[k - N] - a[i] * b[j]) % Q
    return r


def selftest():
    rng = np.random.default_rng(7)
    a = [int(v) for v in rng.integers(0, Q, N)]
    b = [int(v) for v in rng.integers(0, Q, N)]

    ok_rt = invntt(ntt(a)) == a
    print(f"[{'ok ' if ok_rt else 'FAIL'}] invntt(ntt(a)) == a")

    fast = invntt([x * y % Q for x, y in zip(ntt(a), ntt(b))])
    ok_mul = fast == schoolbook(a, b)
    print(f"[{'ok ' if ok_mul else 'FAIL'}] ntt multiply == negacyclic schoolbook")

    ok_root = pow(ZETA, 256, Q) == Q - 1 and pow(ZETA, 512, Q) == 1
    print(f"[{'ok ' if ok_root else 'FAIL'}] zeta is a primitive 512th root of unity")

    good = ok_rt and ok_mul and ok_root
    print("\nNTT core verified" if good else "\nNTT core is WRONG")
    return 0 if good else 1


# ---------------------------------------------------------------------------

LENGTHS = [128, 64, 32, 16, 8, 4, 2, 1]


def butterflies(length):
    """Yield (j, j+length) index pairs for one stage."""
    pairs = []
    start = 0
    while start < N:
        for j in range(start, start + length):
            pairs.append((j, j + length))
        start += 2 * length
    return pairs


def stage_table(vl_values=(2, 4, 8, 16, 32, 64)):
    print(f"{'VL':>4} {'regs':>5}  " + "  ".join(f"s{i}" for i in range(8)) +
          "   cross-lane stages")
    print(f"{'':>4} {'':>5}  " + "  ".join(f"{l:>2}" for l in LENGTHS))
    print("-" * 62)
    for vl in vl_values:
        marks, cross = [], 0
        for length in LENGTHS:
            same = (length % vl == 0)
            marks.append(" ." if same else " X")
            cross += 0 if same else 1
        print(f"{vl:>4} {N // vl:>5}  " + "  ".join(marks) +
              f"   {cross}/8  ({100 * cross / 8:.0f}% need permutes)")
    print("\n  .  = partners share a lane -> plain element-wise vector op")
    print("  X  = partners cross lanes  -> slide / vrgather / scalar fallback")


# ---------------------------------------------------------------------------

class NTTViewer:
    def __init__(self, vl=8):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        self.plt = plt
        self.L3D = Line3DCollection
        self.vl = vl
        self.stage = None          # None = show all
        self.labels = True

        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("Dilithium NTT in vector-register space")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.elev, self.azim = 22, -58
        self.draw()

    def geometry(self):
        """Segments + colors for the current VL and stage filter."""
        segs, cols = [], []
        n_cross = 0
        stages = range(8) if self.stage is None else [self.stage]
        for s in stages:
            length = LENGTHS[s]
            same_lane = (length % self.vl == 0)
            if not same_lane:
                n_cross += 1
            c = "#1D9E75" if same_lane else "#D85A30"
            for j, jl in butterflies(length):
                segs.append([(j % self.vl, j // self.vl, s),
                             (jl % self.vl, jl // self.vl, s)])
                cols.append(c)
        return segs, cols, n_cross

    def draw(self):
        ax = self.ax
        self.elev, self.azim = ax.elev, ax.azim
        ax.clear()

        segs, cols, _ = self.geometry()
        ax.add_collection3d(self.L3D(segs, colors=cols, linewidths=0.55, alpha=0.65))

        nreg = N // self.vl
        xs, ys, zs = [], [], []
        stages = range(8) if self.stage is None else [self.stage]
        for s in stages:
            for i in range(N):
                xs.append(i % self.vl)
                ys.append(i // self.vl)
                zs.append(s)
        ax.scatter(xs, ys, zs, s=2.5, c="#5F5E5A", alpha=0.35,
                   depthshade=False, linewidths=0, marker=".")

        ax.set_xlim(-0.5, max(self.vl - 0.5, 1.5))
        ax.set_ylim(-0.5, nreg - 0.5)
        ax.set_zlim(-0.5, 7.5)
        ax.set_box_aspect((max(self.vl, 2), min(nreg, 24), 9))
        ax.set_xlabel(f"lane  (j mod {self.vl})", labelpad=10)
        ax.set_ylabel(f"vector register  (j div {self.vl})", labelpad=10)
        ax.set_zlabel("NTT stage", labelpad=8)
        ax.set_zticks(range(8))
        if self.labels:
            ax.set_zticklabels([f"{s}  len={LENGTHS[s]}" for s in range(8)], fontsize=8)
        else:
            ax.set_zticklabels([str(s) for s in range(8)])
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.fill = False
            pane.pane.set_edgecolor("#dddddd")

        cross = [l for l in LENGTHS if l % self.vl]
        head = (f"ML-DSA NTT, n=256, q=8380417   |   VL = {self.vl} elements "
                f"({nreg} registers)   |   stage shown: "
                f"{'all' if self.stage is None else self.stage}")
        sub = (f"teal = partners share a lane (element-wise)    "
               f"coral = partners cross lanes (needs permute)\n"
               f"{len(cross)}/8 stages cross lanes"
               + (f"  (len = {', '.join(map(str, cross))})" if cross else ""))
        ax.set_title(head + "\n" + sub, fontsize=10, loc="left", pad=14,
                     family="monospace")
        self.fig.text(0.985, 0.02,
                      "1-6 set VL   up/down isolate stage   s labels   q quit",
                      ha="right", va="bottom", fontsize=8, color="#667788",
                      family="monospace")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = event.key
        vmap = {"1": 2, "2": 4, "3": 8, "4": 16, "5": 32, "6": 64}
        if k in vmap:
            self.vl = vmap[k]
        elif k == "up":
            self.stage = 0 if self.stage is None else min(self.stage + 1, 7)
        elif k == "down":
            if self.stage is None:
                self.stage = 7
            elif self.stage == 0:
                self.stage = None
            else:
                self.stage -= 1
        elif k == "s":
            self.labels = not self.labels
        elif k == "q":
            self.plt.close(self.fig)
            return
        else:
            return
        self.draw()

    def show(self):
        self.plt.show()


def main():
    ap = argparse.ArgumentParser(description="3D view of the ML-DSA NTT in vector-register space.")
    ap.add_argument("--vl", type=int, default=8, help="vector length in 32-bit elements")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.table:
        stage_table()
        return 0

    stage_table((args.vl,))
    print()
    NTTViewer(vl=args.vl).show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
