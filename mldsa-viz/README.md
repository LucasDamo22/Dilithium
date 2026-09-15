# ML-DSA / SHA-3 3D visualizations

Four standalone Python scripts. Only dependencies are numpy and matplotlib:

    pip install numpy matplotlib

Each script runs on its own; there are no imports between them.

## keccak3d.py
The SHA-3 permutation as a live 5x5x64 bit lattice. Steps through all
24 rounds, one step mapping at a time (theta, rho, pi, chi, iota), with a
geometry guide showing the column / lane / slice / row each step acts on.
Press `a` for avalanche mode: bits differing from a run on a 1-bit-flipped
message. Full diffusion lands at round 3.

    python keccak3d.py --selftest        # verify against hashlib
    python keccak3d.py -m "abc"

## ntt3d.py
The ML-DSA NTT butterfly network laid out in vector-register space:
lane on x, register on y, stage on z. Teal butterflies pair coefficients
in the same lane (element-wise vector op), coral ones cross lanes (need a
slide or vrgather). log2(VL) of the 8 stages always cross.

    python ntt3d.py --selftest           # round-trip + schoolbook check
    python ntt3d.py --table              # stage cost table for many VL
    python ntt3d.py --vl 8

## reject3d.py
Rejection sampling as cube containment: an infinity-norm ball is a cube,
so the accept test is "is this point in the box". Second panel derives the
~4.25 expected signing attempts from the per-coefficient bounds.

    python reject3d.py --stats           # numbers only
    python reject3d.py --set 65
    python reject3d.py --beta-scale 1    # true geometry, shell invisible

## dse3d.py
Speedup surface over (vector lanes, Keccak rounds/cycle) with iso-area
curves. THE CONSTANTS ARE A PLACEHOLDER MODEL, NOT MEASUREMENTS -- replace
them with your own profiling and synthesis numbers. The permute penalty on
the NTT side comes from ntt3d.py, so the two stay consistent.

    python dse3d.py --scan               # text sweep
    python dse3d.py --keccak-share 0.55 --area-budget 40000
