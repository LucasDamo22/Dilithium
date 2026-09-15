# Keccak / SHA-3 visualizer — plan and task list

Build order (each stage must actually work before the next starts):

1. **Core + tests** — `keccakviz/core/keccak.py` (Keccak-f[1600] on NumPy uint64
   lanes, batch-capable, full step trace with theta intermediates, constants as
   data), `keccakviz/core/sponge.py` (pad10*1 + domain byte, absorb/squeeze,
   SHA3/SHAKE/Keccak variants, non-standard rate, reduced rounds, step
   toggles). NIST CAVP + hashlib pytest suite. No GUI dependency.
2. **Trace capture / analysis** — `core/trace.py` (snapshot list, JSON export),
   `core/analysis.py` (influence sets per step, avalanche, diffusion heatmap),
   `core/layouts.py` (bit-interleaving, batched lanes).
3. **Minimal 3D view** — ModernGL instanced cubes inside a `QOpenGLWidget`,
   orbit/pan/zoom camera, presets, color modes, step/round/play, animated
   transitions for rho and pi, feed-highlighting for theta and chi, picking.
4. **Other modes** — slice stack, lane table, raw bytes, diffusion heatmap,
   avalanche view + Hamming plot, bit-interleaved, batched lanes.
5. **Sponge screen** — message input, padding, rate/capacity bar, timeline,
   permutation counter, click-to-jump.
6. **Parameters** — variant, domain byte, rate/capacity, rounds, output
   length, message, step toggles; all live.
7. **Explanation panel** — Plain / Student / Expert, tracks current step.
8. Export (PNG, JSON), keyboard shortcuts, profiling, README, self-review.

## Task list

- [x] venv + deps (numpy, moderngl, PyQt5, pytest)
- [x] NIST CAVP vectors vendored under `tests/kat`
- [x] core/keccak.py with trace
- [x] core/sponge.py with variants and padding
- [x] tests green (KAT + hashlib + structural)
- [x] trace JSON export
- [x] analysis: influence, avalanche, diffusion
- [x] layouts: interleave, batched
- [x] GL cube view (instanced), camera, picking
- [x] animations: rho, pi, theta, chi, iota
- [x] transport: step/round/play/speed/keys
- [x] 2D modes
- [x] sponge screen
- [x] parameter panel
- [x] explanation panel
- [x] export PNG/JSON
- [x] profile 1600-cell animation
- [x] README
- [x] self-review, KATs still green
