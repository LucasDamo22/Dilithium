#!/usr/bin/env python3
"""
dse3d.py -- design-space surface for the ML-DSA partitioning question:
given a fixed area budget, how much goes to vector lanes (which accelerate
the NTT) and how much to a dedicated Keccak unit (which accelerates
SHAKE, and which Zve32x cannot help with at all because ELEN=32)?

    x axis = vector lanes           (elements per vector op)
    y axis = Keccak rounds / cycle  (how hard the Keccak unit is unrolled)
    z axis = signing speedup over the scalar baseline

The left panel is the speedup surface. The right panel is the same surface
from above, with ISO-AREA curves drawn on it -- the design question is
where a given area curve touches the highest speedup contour, and that
point is marked.

>>> THE CONSTANTS BELOW ARE A PLACEHOLDER MODEL, NOT MEASUREMENTS. <<<
Replace them with your own RS5 profiling and synthesis numbers before
putting any of this in a paper. What the script is actually for is the
SHAPE of the tradeoff, which is robust to the exact values: the NTT side
saturates (Amdahl plus a permute penalty that grows with lane count) while
the Keccak side keeps paying off, because Keccak is the larger share and
has no vector path.

The permute penalty is not invented -- it is the cross-lane stage count
from ntt3d.py: an NTT of length 256 has 8 stages, and log2(VL) of them
pair coefficients that live in different lanes of the same register.

Requires: numpy, matplotlib  ->  pip install numpy matplotlib

Usage:
    python dse3d.py
    python dse3d.py --keccak-share 0.55 --area-budget 40000
    python dse3d.py --scan            # text sweep, no GUI
"""

import argparse
import sys

import numpy as np

# --- placeholder workload model -------------------------------------------
KECCAK_SHARE = 0.65     # fraction of scalar signing cycles inside Keccak-f
NTT_SHARE = 0.20        # fraction inside NTT / pointwise / invNTT
# remainder is scalar glue: packing, decompose, hint generation, control

SCALAR_ROUNDS_PER_CYCLE = 1 / 24.0   # software Keccak: ~1 permutation per 24+ "rounds"
PERMUTE_PENALTY = 2.5   # a cross-lane stage costs this much vs an in-lane stage

# --- placeholder area model (arbitrary gate-equivalent-ish units) ----------
AREA_BASE = 12000.0     # RS5 core without the accelerators
AREA_PER_LANE = 1400.0  # one more 32-bit vector lane + regfile ports
AREA_KECCAK = 9000.0    # a Keccak unit doing 1 round/cycle
# ---------------------------------------------------------------------------


def ntt_efficiency(lanes):
    """Effective speedup of the NTT given cross-lane stages cost more."""
    lanes = np.asarray(lanes, dtype=float)
    cross = np.log2(np.maximum(lanes, 1.0))          # stages needing permutes
    cross = np.minimum(cross, 8.0)
    weighted = (8.0 - cross) + cross * PERMUTE_PENALTY
    return lanes * (8.0 / weighted)


def speedup(lanes, krate, keccak_share=KECCAK_SHARE, ntt_share=NTT_SHARE):
    other = 1.0 - keccak_share - ntt_share
    t = (keccak_share * (SCALAR_ROUNDS_PER_CYCLE / np.maximum(krate, 1e-9))
         + ntt_share / ntt_efficiency(lanes)
         + other)
    return 1.0 / t


def area(lanes, krate):
    return AREA_BASE + AREA_PER_LANE * np.asarray(lanes, float) \
        + AREA_KECCAK * np.asarray(krate, float)


def scan(keccak_share, ntt_share):
    print(f"model: keccak {keccak_share:.0%}, ntt {ntt_share:.0%}, "
          f"other {1 - keccak_share - ntt_share:.0%}   (PLACEHOLDER NUMBERS)\n")
    print(f"{'lanes':>6} {'eff':>6} | " +
          " ".join(f"k={k:<4g}" for k in (0.25, 0.5, 1, 2, 4)))
    print("-" * 52)
    for L in (1, 2, 4, 8, 16, 32):
        row = " ".join(f"{speedup(L, k):>6.2f}" for k in (0.25, 0.5, 1, 2, 4))
        print(f"{L:>6} {ntt_efficiency(L):>6.2f} | {row}")
    print("\n'eff' = effective NTT speedup after the cross-lane permute penalty.")
    print("Note how the columns move far more than the rows: with Keccak at")
    print(f"{keccak_share:.0%} of the profile and no vector path, lanes alone cannot get you far.")


def main():
    ap = argparse.ArgumentParser(description="Area-vs-speedup surface for an ML-DSA accelerator split.")
    ap.add_argument("--keccak-share", type=float, default=KECCAK_SHARE)
    ap.add_argument("--ntt-share", type=float, default=NTT_SHARE)
    ap.add_argument("--area-budget", type=float, default=None,
                    help="draw the optimum on this iso-area curve")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    ks, ns = args.keccak_share, args.ntt_share
    if ks + ns >= 1.0:
        print("keccak-share + ntt-share must be < 1", file=sys.stderr)
        return 1

    scan(ks, ns)
    if args.scan:
        return 0

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required:  pip install numpy matplotlib", file=sys.stderr)
        return 1

    L = np.linspace(1, 16, 90)
    K = np.linspace(0.125, 4.0, 90)
    LG, KG = np.meshgrid(L, K)
    S = speedup(LG, KG, ks, ns)
    A = area(LG, KG)

    budget = args.area_budget if args.area_budget else float(np.median(A))

    fig = plt.figure(figsize=(13.5, 6))
    fig.canvas.manager.set_window_title("ML-DSA accelerator design space")

    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(LG, KG, S, cmap="viridis", linewidth=0,
                    antialiased=True, alpha=0.92, rstride=2, cstride=2)
    ax.contour(LG, KG, S, levels=10, zdir="z",
               offset=float(S.min()), colors="#888780", linewidths=0.4)
    ax.set_xlabel("vector lanes", labelpad=8)
    ax.set_ylabel("Keccak rounds / cycle", labelpad=8)
    ax.set_zlabel("signing speedup", labelpad=8)
    ax.set_title(f"speedup surface\nkeccak {ks:.0%} / ntt {ns:.0%} of scalar profile "
                 f"(placeholder model)", fontsize=9, loc="left",
                 family="monospace", pad=10)
    ax.view_init(elev=24, azim=-131)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("#dddddd")

    ax2 = fig.add_subplot(122)
    cf = ax2.contourf(LG, KG, S, levels=18, cmap="viridis", alpha=0.9)
    fig.colorbar(cf, ax=ax2, label="signing speedup", shrink=0.85)
    ca = ax2.contour(LG, KG, A, levels=6, colors="#ffffff",
                     linewidths=1.0, linestyles="--")
    ax2.clabel(ca, fmt="%.0f", fontsize=7, colors="#ffffff")

    on = np.abs(A - budget) < (AREA_PER_LANE * 0.35)
    if on.any():
        best = np.argmax(np.where(on, S, -np.inf))
        by, bx = np.unravel_index(best, S.shape)
        ax2.plot(LG[by, bx], KG[by, bx], "o", ms=10, mfc="none",
                 mec="#D85A30", mew=2.2)
        ax2.annotate(f" {LG[by, bx]:.0f} lanes, {KG[by, bx]:.2f} rounds/cyc\n"
                     f" speedup {S[by, bx]:.2f}x at area {budget:.0f}",
                     xy=(LG[by, bx], KG[by, bx]), xytext=(1.6, 3.35),
                     fontsize=8.5, family="monospace", color="#ffffff")

    ax2.set_xlabel("vector lanes")
    ax2.set_ylabel("Keccak rounds / cycle")
    ax2.set_title("dashed = iso-area, circle = best point on the chosen budget",
                  fontsize=9, loc="left", family="monospace", pad=10)

    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
