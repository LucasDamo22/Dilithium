#!/usr/bin/env python3
"""
inverse_clock.py -- the modular inverse as a walk around a clock.

Put a hand on 0 of an m-hour clock and move it forward a hours at a time.
a^-1 mod m is the number of hops it takes to land on 1. Every time the hand
passes 0 it has done one full lap, so landing on 1 after x hops means

    a*x = laps*m + 1        i.e.   a*x = 1 (mod m)

The hop count x IS the inverse, and (x, -laps) is a Bezout pair for a and m:
the same a*x + m*y = 1 that egcd solves. egcd may hand back a different
pair, but its x reduced mod m is always the hop count.

If gcd(a, m) = g > 1 the hand only ever lands on multiples of g. It returns
to 0 after m/g hops without touching 1: there is no inverse.

Powers mode (p) draws the Fermat route instead: hop from a^k to a^(k+1).
For prime m, a^(m-1) = 1, so the stop just before it, a^(m-2), is the
inverse.

The side panel shows the same inverse coming out of extended Euclid, with
the variable names from modarith.py (dividend, divisor, quotient,
remainder) and the (x, y) recipe carried beside each number.

The spiral is drawn with the hand moving inward a little on every hop, so
laps that pass over the same hours stay visible as separate rings.

Requires: matplotlib  ->  pip install matplotlib

Usage:
    python inverse_clock.py                  # a = 7, m = 17
    python inverse_clock.py --a 4 --m 10     # no inverse: the walk never hits 1
    python inverse_clock.py --mode powers    # the Fermat route
    python inverse_clock.py --text           # ledger + egcd trace, no GUI
    python inverse_clock.py --selftest       # check the three methods agree
    python inverse_clock.py --save out.png   # render one frame, no window

Keys:  right / left  -> one hop forward / back     enter -> all hops
       up / down     -> a + 1 / a - 1              ] [   -> m + 1 / m - 1
       p             -> hops <-> powers            r     -> back to 0 hops
       q             -> quit
"""

import argparse
import math
import sys

TEAL = "#1D9E75"
CORAL = "#D85A30"
GREY = "#5F5E5A"
FAINT = "#d6d6d2"
INK = "#2C2C2A"

M_MIN, M_MAX = 2, 97
LEDGER_ROWS = 13


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def hop_walk(a, m):
    """Walk 0, a, 2a, ... mod m until the hand lands on 1 or comes back to 0.

    Returns (positions, inverse). positions[k] is where hop k lands, with
    positions[0] = 0 the start. inverse is the hop count that reached 1, or
    None if the hand got back to 0 first.
    """
    positions = [0]
    for k in range(1, m + 1):
        positions.append(a * k % m)
        if positions[-1] == 1:
            return positions, k
        if positions[-1] == 0:
            return positions, None
    return positions, None


def power_walk(a, m):
    """a^0, a^1, ..., a^(m-1) mod m. For prime m the last one is 1 (Fermat)."""
    return [pow(a, k, m) for k in range(m)]


def egcd_trace(a, b):
    """Extended Euclid using the variable names from modarith.py.

    Every number carries its recipe (x, y): number = x*a + y*b.
    Returns (rows, g, x, y), one row per pass of the loop.
    """
    if b >= a:
        dividend, divisor = b, a
        dividend_x, dividend_y, divisor_x, divisor_y = 0, 1, 1, 0
    else:
        dividend, divisor = a, b
        dividend_x, dividend_y, divisor_x, divisor_y = 1, 0, 0, 1
    rows = []
    while True:
        quotient, remainder = divmod(dividend, divisor)
        remainder_x = dividend_x - quotient * divisor_x
        remainder_y = dividend_y - quotient * divisor_y
        rows.append((dividend, dividend_x, dividend_y,
                     divisor, divisor_x, divisor_y,
                     quotient, remainder, remainder_x, remainder_y))
        if remainder == 0:
            return rows, divisor, divisor_x, divisor_y
        dividend, dividend_x, dividend_y = divisor, divisor_x, divisor_y
        divisor, divisor_x, divisor_y = remainder, remainder_x, remainder_y


# ---------------------------------------------------------------------------
# Text blocks. Each is a list of (line, colour) so the GUI panel and --text
# print exactly the same thing.


def hop_block(a, m, shown=None, window=None):
    positions, inv = hop_walk(a, m)
    n = len(positions) - 1
    shown = n if shown is None else min(shown, n)
    out = [(f"HOPS   start at 0, move {a} hours per hop, wrap at {m}", INK),
           (f"  hop   total =  laps*{m} + lands on", GREY)]
    first = 1 if window is None else max(1, shown - window + 1)
    if first > 1:
        out.append((f"   ... hops 1-{first - 1} above", GREY))
    for k in range(first, shown + 1):
        laps, pos = divmod(a * k, m)
        note, colour = "", INK
        if pos == 1:
            note, colour = "   <- landed on 1", CORAL
        elif pos == 0:
            note, colour = "   <- back at 0", GREY
        out.append((f"  {k:>3}  {a * k:>6} = {laps:>4}*{m} + {pos:>3}{note}", colour))
    if shown < n:
        out.append((f"   ... {n - shown} more hop(s): right / enter", GREY))
    elif inv is not None:
        laps = (a * inv) // m
        out.append((f"=> {a}*{inv} = {a * inv} ≡ 1 (mod {m}),  so {a}^-1 = {inv}", CORAL))
        out.append((f"   {inv} hops, {laps} full laps:  {a}*{inv} + {m}*({-laps}) = 1", INK))
    else:
        g = math.gcd(a, m)
        out.append((f"=> back at 0 after {n} hops, never on 1", CORAL))
        out.append((f"   the hand only visits multiples of gcd = {g}: no inverse", INK))
    return out


def power_block(a, m, shown=None, window=None):
    vals = power_walk(a, m)
    n = m - 1
    shown = n if shown is None else min(shown, n)
    prime = is_prime(m)
    head = "prime, Fermat applies" if prime else "NOT prime, Fermat does not apply"
    out = [(f"POWERS   a^k mod {m}   (m = {m}: {head})", INK),
           (f"    k   {a}^k mod {m}", GREY)]
    first = 0 if window is None else max(0, shown - window + 1)
    if first > 0:
        out.append((f"   ... k = 0-{first - 1} above", GREY))
    for k in range(first, shown + 1):
        note, colour = "", INK
        if k == m - 2:
            note, colour = "   <- a^(m-2)", CORAL
        elif k == m - 1:
            note = "   <- a^(m-1)"
        out.append((f"  {k:>3}   {vals[k]:>4}{note}", colour))
    if shown < n:
        out.append((f"   ... {n - shown} more step(s): right / enter", GREY))
    elif m >= 3:
        cand = vals[m - 2]
        check = a * cand % m
        if check == 1:
            out.append((f"=> {a}^{m - 2} ≡ {cand},  {a}*{cand} ≡ 1 (mod {m}),  so {a}^-1 = {cand}", CORAL))
        else:
            out.append((f"=> {a}^{m - 2} ≡ {cand}, but {a}*{cand} ≡ {check} (mod {m}): not the inverse", CORAL))
        if prime:
            out.append((f"   a^{m - 1} = {vals[m - 1]}: the orbit closes on 1, as Fermat says", INK))
        else:
            out.append(("   Fermat's a^(m-1) = 1 needs m prime; use egcd below", INK))
    return out


def egcd_block(a, m):
    rows, g, x, y = egcd_trace(a, m)
    out = [(f"EGCD   egcd({a}, {m}), names from modarith.py", INK),
           ("  dividend (x, y)   divisor (x, y)   quot   remainder (x, y)", GREY)]
    for (dv, dx, dy, ds, sx, sy, q, r, rx, ry) in rows:
        rem = f"{r:>4} ({rx:>3},{ry:>3})" if r else "   0  -> break"
        out.append((f"  {dv:>4} ({dx:>3},{dy:>3})   {ds:>4} ({sx:>3},{sy:>3})   {q:>4}   {rem}", INK))
    out.append((f"=> (g, x, y) = ({g}, {x}, {y})    {a}*({x}) + {m}*({y}) = {g}", INK))
    if g == 1:
        out.append((f"   x % {m} = {x % m}   <- the inverse, same as the clock", CORAL))
    else:
        out.append((f"   g = {g}, not 1: no inverse (modinv raises ValueError)", CORAL))
    return out


# ---------------------------------------------------------------------------


def selftest():
    hop_bad = none_bad = egcd_bad = fermat_bad = 0
    pairs = 0
    for m in range(M_MIN, 200):
        for a in range(1, m):
            pairs += 1
            _, inv = hop_walk(a, m)
            _, g, x, y = egcd_trace(a, m)
            if g != math.gcd(a, m) or a * x + m * y != g:
                egcd_bad += 1
            if g == 1:
                ref = pow(a, -1, m)
                if inv != ref:
                    hop_bad += 1
                if x % m != ref:
                    egcd_bad += 1
                if is_prime(m) and pow(a, m - 2, m) != ref:
                    fermat_bad += 1
            elif inv is not None:
                none_bad += 1

    def line(bad, what):
        print(f"[{'ok ' if bad == 0 else 'FAIL'}] {what}" + (f"   ({bad} wrong)" if bad else ""))
        return bad == 0

    print(f"checked every a in [1, m) for 2 <= m < 200  ({pairs} pairs)\n")
    good = all([
        line(hop_bad, "hop count to reach 1 == pow(a, -1, m) whenever gcd(a, m) = 1"),
        line(none_bad, "the walk never reaches 1 when gcd(a, m) > 1"),
        line(egcd_bad, "egcd: g == gcd, a*x + m*y == g, and x % m == the inverse"),
        line(fermat_bad, "Fermat: a^(m-2) == the inverse for every prime m"),
    ])
    print("\nall three methods agree" if good else "\nMETHODS DISAGREE")
    return 0 if good else 1


def text_report(a, m):
    blocks = [hop_block(a, m), egcd_block(a, m)]
    if is_prime(m):
        blocks.insert(1, power_block(a, m))
    print("\n\n".join("\n".join(line for line, _ in b) for b in blocks))


# ---------------------------------------------------------------------------


def angle(t, m):
    """Clock angle for t hours: 0 at the top, running clockwise."""
    return math.pi / 2 - 2 * math.pi * t / m


def xy(t, m, r=1.0):
    th = angle(t, m)
    return r * math.cos(th), r * math.sin(th)


class ClockViewer:
    def __init__(self, a=7, m=17, mode="hops", headless=False):
        import matplotlib
        if headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # free up keys that matplotlib binds by default
        for name, keys in (("keymap.back", ["left"]), ("keymap.forward", ["right"]),
                           ("keymap.pan", ["p"]), ("keymap.home", ["r", "home"])):
            plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in keys]

        self.plt, self.np = plt, np
        self.m = max(M_MIN, min(M_MAX, m))
        self.a = a % self.m or 1
        self.mode = mode
        self.step = self.total_steps()

        self.fig = plt.figure(figsize=(14, 7.8))
        self.fig.canvas.manager.set_window_title("Modular inverse on a clock")
        self.ax = self.fig.add_axes([0.01, 0.06, 0.49, 0.80])
        self.side = self.fig.add_axes([0.52, 0.06, 0.47, 0.80])
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def total_steps(self):
        if self.mode == "hops":
            return len(hop_walk(self.a, self.m)[0]) - 1
        return self.m - 1

    # -- clock face ---------------------------------------------------------

    def draw_face(self, visited, target, highlight):
        ax, np, m = self.ax, self.np, self.m
        t = np.linspace(0, m, 600)
        th = np.pi / 2 - 2 * np.pi * t / m
        ax.plot(np.cos(th), np.sin(th), color=GREY, lw=1.0, alpha=0.55)

        size = 12 if m <= 24 else 9.5 if m <= 48 else 7
        for p in range(m):
            (x0, y0), (x1, y1) = xy(p, m, 0.965), xy(p, m, 1.0)
            ax.plot([x0, x1], [y0, y1], color=GREY, lw=0.8)
            labelled = m <= 48 or p % 5 == 0 or p in (target, highlight) or p in visited
            if not labelled:
                continue
            if p == target or p == highlight:
                colour, weight = CORAL, "bold"
            elif p in visited:
                colour, weight = TEAL, "bold"
            else:
                colour, weight = "#9a9a95", "normal"
            lx, ly = xy(p, m, 1.1)
            ax.text(lx, ly, str(p), ha="center", va="center", fontsize=size,
                    color=colour, weight=weight, family="monospace")

        for p in visited:
            ax.plot(*xy(p, m, 1.0), "o", ms=5, color=TEAL, zorder=4)
        if target is not None:
            ax.plot(*xy(target, m, 1.0), "o", ms=13, mfc="none", mec=CORAL, mew=1.8, zorder=4)

    # -- hops mode ----------------------------------------------------------

    def draw_hops(self):
        ax, np, a, m = self.ax, self.np, self.a, self.m
        positions, inv = hop_walk(a, m)
        n = len(positions) - 1
        k_now = self.step
        visited = set(positions[1:k_now + 1])
        self.draw_face(visited, target=1, highlight=None)

        r0 = 0.88
        dr = min(0.06, 0.58 / max(n, 1))
        hours = a * k_now
        if hours:
            t = np.linspace(0, hours, max(80, min(hours * 14, 20000)))
            r = r0 - dr * t / a
            th = np.pi / 2 - 2 * np.pi * t / m
            ax.plot(r * np.cos(th), r * np.sin(th), color=TEAL, lw=1.7, alpha=0.9, zorder=3)
        ax.plot(*xy(0, m, r0), marker="s", ms=6, color=INK, zorder=5)

        label_all = n <= 30
        for k in range(1, k_now + 1):
            pos, rk = positions[k], r0 - dr * k
            last = k == k_now
            colour = CORAL if pos == 1 else GREY if pos == 0 else TEAL
            ax.plot(*xy(pos, m, rk), "o", ms=8 if last else 4.5, color=colour, zorder=6)
            if label_all or last:
                lx, ly = xy(pos, m, rk - 0.06)
                ax.text(lx, ly, str(k), ha="center", va="center", fontsize=7.5,
                        color=INK, zorder=7)
            if last:
                (x0, y0), (x1, y1) = xy(pos, m, rk), xy(pos, m, 0.965)
                ax.plot([x0, x1], [y0, y1], ls="--", lw=0.9, color=colour, zorder=2)

        if k_now == 0:
            centre, colour = f"press right:\nhop {a} hours", GREY
        elif k_now < n:
            centre, colour = f"hop {k_now}\n{a}*{k_now} = {a * k_now}\nlands on {positions[k_now]}", INK
        elif inv is not None:
            centre, colour = f"{a}^-1 = {inv}\n({inv} hops)", CORAL
        else:
            centre, colour = f"no inverse\ngcd = {math.gcd(a, m)}", CORAL
        ax.text(0, 0, centre, ha="center", va="center", fontsize=12, color=colour,
                family="monospace", weight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85))

    # -- powers mode --------------------------------------------------------

    def draw_powers(self):
        ax, a, m = self.ax, self.a, self.m
        vals = power_walk(a, m)
        k_now = self.step
        inv_k = m - 2
        highlight = vals[inv_k] if k_now >= inv_k and m >= 3 else None
        visited = set(vals[:k_now + 1])
        self.draw_face(visited, target=1, highlight=highlight)

        r = 0.86
        for k in range(1, k_now + 1):
            p0, p1 = vals[k - 1], vals[k]
            if p0 == p1:
                continue
            colour = CORAL if k == m - 1 else TEAL
            ax.annotate("", xy=xy(p1, m, r), xytext=xy(p0, m, r), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1,
                                        alpha=0.8, shrinkA=5, shrinkB=5, mutation_scale=11))

        exps = {}
        for k in range(k_now + 1):
            exps.setdefault(vals[k], []).append(k)
        for p, ks in exps.items():
            colour = CORAL if p == highlight else TEAL
            ax.plot(*xy(p, m, r), "o", ms=6, color=colour, zorder=6)
            if m <= 48:
                txt = ",".join(map(str, ks[:3])) + ("..." if len(ks) > 3 else "")
                lx, ly = xy(p, m, r - 0.085)
                ax.text(lx, ly, txt, ha="center", va="center", fontsize=7, color=INK, zorder=7)

        if k_now < m - 1:
            centre, colour = f"k = {k_now}\n{a}^{k_now} = {vals[k_now]}", INK
        elif not is_prime(m):
            centre, colour = f"m = {m} not prime\nFermat fails", CORAL
        else:
            centre, colour = f"{a}^{m - 2} = {vals[m - 2]}\n= {a}^-1", CORAL
        ax.text(0, 0, centre, ha="center", va="center", fontsize=12, color=colour,
                family="monospace", weight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85))

    # -- side panel ---------------------------------------------------------

    def draw_side(self):
        side, a, m = self.side, self.a, self.m
        if self.mode == "hops":
            lines = hop_block(a, m, self.step, LEDGER_ROWS)
        else:
            lines = power_block(a, m, self.step, LEDGER_ROWS)
        lines = lines + [("", INK)] + egcd_block(a, m)

        height_in = self.fig.get_figheight() * 0.80
        step = 9 * 1.35 / 72 / height_in
        for i, (text, colour) in enumerate(lines):
            side.text(0.0, 1.0 - i * step, text, transform=side.transAxes, va="top",
                      ha="left", fontsize=9, family="monospace", color=colour)

    # -- frame --------------------------------------------------------------

    def draw(self):
        for ax in (self.ax, self.side):
            ax.clear()
            ax.axis("off")
        for t in list(self.fig.texts):
            t.remove()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-1.22, 1.22)
        self.ax.set_ylim(-1.22, 1.22)

        if self.mode == "hops":
            self.draw_hops()
            legend = ("teal = where the hand lands    coral ring = 1, the target    "
                      "grey = back at 0 (no inverse)")
        else:
            self.draw_powers()
            legend = ("teal = a^k    coral = a^(m-2), the Fermat inverse    "
                      "coral ring = 1, where a^(m-1) must land")
        self.draw_side()

        g = math.gcd(self.a, self.m)
        head = (f"Modular inverse on a clock   |   a = {self.a}, m = {self.m}"
                f"{' (prime)' if is_prime(self.m) else ''}   |   gcd = {g}   |   "
                f"mode: {self.mode}   |   step {self.step}/{self.total_steps()}")
        self.fig.text(0.015, 0.965, head, fontsize=10.5, family="monospace", color=INK, va="top")
        self.fig.text(0.015, 0.93, legend, fontsize=9, family="monospace", color=GREY, va="top")
        self.fig.text(0.985, 0.015,
                      "right/left hop   enter all   r reset   up/down a   ] [ m   p mode   q quit",
                      ha="right", va="bottom", fontsize=8, color="#667788", family="monospace")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = event.key
        if k == "right":
            self.step = min(self.step + 1, self.total_steps())
        elif k == "left":
            self.step = max(self.step - 1, 0)
        elif k == "enter":
            self.step = self.total_steps()
        elif k == "r":
            self.step = 0
        elif k in ("up", "down"):
            self.a = (self.a - 1 + (1 if k == "up" else -1)) % (self.m - 1) + 1
            self.step = self.total_steps()
        elif k in ("]", "["):
            self.m = max(M_MIN, min(M_MAX, self.m + (1 if k == "]" else -1)))
            self.a = self.a % self.m or 1
            self.step = self.total_steps()
        elif k == "p":
            self.mode = "powers" if self.mode == "hops" else "hops"
            self.step = self.total_steps()
        elif k == "q":
            self.plt.close(self.fig)
            return
        else:
            return
        self.draw()

    def show(self):
        self.plt.show()

    def save(self, path):
        self.fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description="The modular inverse as a walk around a clock.")
    ap.add_argument("--a", type=int, default=7, help="the number to invert (default 7)")
    ap.add_argument("--m", type=int, default=17, help=f"the modulus, {M_MIN}..{M_MAX} (default 17)")
    ap.add_argument("--mode", choices=("hops", "powers"), default="hops")
    ap.add_argument("--step", type=int, help="start at this hop instead of showing all of them")
    ap.add_argument("--text", action="store_true", help="print the ledger and egcd trace, no GUI")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--save", metavar="FILE", help="render one frame to an image, no window")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not M_MIN <= args.m <= M_MAX:
        ap.error(f"--m must be between {M_MIN} and {M_MAX}")
    if args.a % args.m == 0:
        ap.error("--a must not be a multiple of m (0 never has an inverse)")
    if args.text:
        text_report(args.a % args.m, args.m)
        return 0

    viewer = ClockViewer(args.a, args.m, args.mode, headless=bool(args.save))
    if args.step is not None:
        viewer.step = max(0, min(args.step, viewer.total_steps()))
        viewer.draw()
    if args.save:
        viewer.save(args.save)
        print(f"wrote {args.save}")
        return 0
    viewer.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
