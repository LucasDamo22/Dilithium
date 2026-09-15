#!/usr/bin/env python3
"""
reject3d.py -- the geometry of ML-DSA (Dilithium) rejection sampling.

Signing computes z = y + c*s1 and throws the whole signature away unless
every coefficient satisfies an infinity-norm bound. An infinity-norm ball
is a CUBE, so the accept/reject test is literally "is this point inside the
box" -- which is why 3D is the right picture for it.

The left panel plots triples of z coefficients against two nested cubes:

    outer cube  |z| <= gamma1        the range y was sampled from
    inner cube  |z| <  gamma1 - beta the acceptance region

The shell between them is the rejection region. At true ML-DSA-44
parameters that shell is 0.06% of the half-width -- invisible. So the
drawing inflates beta by --beta-scale purely for visibility; every number
reported is computed at the real parameters.

The right panel is the part that actually explains the 4.25 iterations:
per-coefficient acceptance is 99.94%, but it is raised to the power of the
coefficient count, and 1024 coefficients is a lot of power.

Requires: numpy, matplotlib  ->  pip install numpy matplotlib

Usage:
    python reject3d.py
    python reject3d.py --set 65        # ML-DSA-65
    python reject3d.py --beta-scale 1  # true geometry (shell invisible)
    python reject3d.py --stats         # numbers only, no GUI
"""

import argparse
import sys

import numpy as np

Q = 8380417

# name: (k, l, eta, tau, gamma1, gamma2, omega)
PARAMS = {
    44: dict(k=4, l=4, eta=2, tau=39, gamma1=1 << 17, gamma2=(Q - 1) // 88, omega=80),
    65: dict(k=6, l=5, eta=4, tau=49, gamma1=1 << 19, gamma2=(Q - 1) // 32, omega=55),
    87: dict(k=8, l=7, eta=2, tau=60, gamma1=1 << 19, gamma2=(Q - 1) // 32, omega=75),
}


def analyse(name):
    p = PARAMS[name]
    beta = p["tau"] * p["eta"]
    n_z = 256 * p["l"]                 # coefficients checked by the z bound
    n_r = 256 * p["k"]                 # coefficients checked by the r0 bound

    pz = 1.0 - beta / p["gamma1"]      # per-coefficient, ||z||inf < gamma1 - beta
    pr = 1.0 - beta / p["gamma2"]      # per-coefficient, ||r0||inf < gamma2 - beta
    Pz, Pr = pz ** n_z, pr ** n_r
    P = Pz * Pr
    return dict(p, name=name, beta=beta, n_z=n_z, n_r=n_r,
                pz=pz, pr=pr, Pz=Pz, Pr=Pr, P=P, iters=1.0 / P)


def print_stats(a):
    print(f"ML-DSA-{a['name']}   k={a['k']} l={a['l']} eta={a['eta']} tau={a['tau']} "
          f"omega={a['omega']}")
    print(f"  gamma1 = {a['gamma1']:>8}    gamma2 = {a['gamma2']:>8}    "
          f"beta = tau*eta = {a['beta']}")
    print()
    print(f"  per-coefficient  P(|z|  < gamma1-beta) = 1 - {a['beta']}/{a['gamma1']} "
          f"= {a['pz']:.6f}")
    print(f"  per-coefficient  P(|r0| < gamma2-beta) = 1 - {a['beta']}/{a['gamma2']} "
          f"= {a['pr']:.6f}")
    print()
    print(f"  over {a['n_z']:>4} z coefficients   -> {a['Pz']:.4f}")
    print(f"  over {a['n_r']:>4} r0 coefficients  -> {a['Pr']:.4f}")
    print(f"  both                      -> {a['P']:.4f}")
    print()
    print(f"  expected signing attempts = 1/{a['P']:.4f} = {a['iters']:.2f}")
    print(f"  (the hint-weight check |h| <= omega trims this slightly further)")


def sample_z(a, rng, n):
    """n triples of z coefficients at the real parameters: z = y + c*s1."""
    g1 = a["gamma1"]
    y = rng.integers(-g1 + 1, g1 + 1, size=(n, 3))
    # c*s1 coefficient: sum of tau signed eta-bounded terms, bounded by beta.
    cs1 = np.clip(rng.normal(0, a["beta"] / 3.0, size=(n, 3)).round(),
                  -a["beta"], a["beta"])
    return y + cs1


def cube_edges(h):
    v = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
         (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [[v[i], v[j]] for i, j in e]


def main():
    ap = argparse.ArgumentParser(description="3D geometry of ML-DSA rejection sampling.")
    ap.add_argument("--set", type=int, default=44, choices=(44, 65, 87))
    ap.add_argument("--points", type=int, default=2500)
    ap.add_argument("--beta-scale", type=float, default=300.0,
                    help="inflate beta in the DRAWING so the shell is visible")
    ap.add_argument("--stats", action="store_true", help="numbers only, no GUI")
    args = ap.parse_args()

    a = analyse(args.set)
    print_stats(a)
    if args.stats:
        return 0

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError:
        print("matplotlib is required:  pip install numpy matplotlib", file=sys.stderr)
        return 1

    rng = np.random.default_rng(1)
    z = sample_z(a, rng, args.points)

    g1 = a["gamma1"]
    beta_draw = min(a["beta"] * args.beta_scale, g1 * 0.35)
    inner = g1 - beta_draw
    inside = np.all(np.abs(z) < inner, axis=1)

    fig = plt.figure(figsize=(13, 6))
    fig.canvas.manager.set_window_title(f"ML-DSA-{args.set} rejection geometry")

    ax = fig.add_subplot(121, projection="3d")
    ax.add_collection3d(Line3DCollection(cube_edges(g1), colors="#888780",
                                         linewidths=1.0, alpha=0.8))
    ax.add_collection3d(Line3DCollection(cube_edges(inner), colors="#1D9E75",
                                         linewidths=1.6, alpha=0.95))
    ax.scatter(*z[inside].T, s=5, c="#1D9E75", alpha=0.35,
               depthshade=False, linewidths=0)
    ax.scatter(*z[~inside].T, s=14, c="#D85A30", alpha=0.9,
               depthshade=False, linewidths=0)
    lim = g1 * 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]"); ax.set_zlabel("z[2]")
    ax.set_title(
        f"three coefficients of z = y + c*s1\n"
        f"grey cube |z| <= gamma1 = {g1}\n"
        f"green cube = acceptance region   (beta inflated {args.beta_scale:g}x to be visible)",
        fontsize=9, loc="left", family="monospace", pad=12)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("#dddddd")
    ax.view_init(elev=20, azim=-55)

    ax2 = fig.add_subplot(122)
    ns = np.arange(1, max(a["n_z"], a["n_r"]) + 1)
    ax2.plot(ns, a["pz"] ** ns, color="#378ADD", lw=1.6,
             label=f"$\\|z\\|_\\infty$ bound  (per-coeff {a['pz']:.5f})")
    ax2.plot(ns, a["pr"] ** ns, color="#BA7517", lw=1.6,
             label=f"$\\|r_0\\|_\\infty$ bound  (per-coeff {a['pr']:.5f})")
    ax2.plot(ns, (a["pz"] ** np.minimum(ns, a["n_z"])) *
             (a["pr"] ** np.minimum(ns, a["n_r"])),
             color="#534AB7", lw=2.2, label="both")
    ax2.axvline(a["n_z"], color="#888780", ls=":", lw=1)
    ax2.plot([a["n_z"]], [a["P"]], "o", color="#D85A30", ms=7, zorder=5)
    ax2.annotate(f"  {a['n_z']} coeffs -> P = {a['P']:.3f}\n"
                 f"  {a['iters']:.2f} attempts per signature",
                 xy=(a["n_z"], a["P"]), xytext=(a["n_z"] * 0.42, a["P"] + 0.16),
                 fontsize=9, family="monospace", color="#4A1B0C")
    ax2.set_xlabel("number of coefficients checked")
    ax2.set_ylabel("probability all pass")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlim(0, ns[-1])
    ax2.set_title("why a 99.94% test fails half the time", fontsize=10,
                  loc="left", family="monospace", pad=10)
    ax2.legend(fontsize=8, loc="upper right", frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(alpha=0.2, lw=0.5)

    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
