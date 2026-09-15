"""GUI-free tests of the UI's math: animation frames, camera, session navigation."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from keccakviz.core import keccak as K
from keccakviz.ui import anim
from keccakviz.ui.camera import OrbitCamera


def bits_of(state):
    return K.lanes_to_bits(state).reshape(-1)


@pytest.fixture(scope="module")
def trace():
    s = K.state_from_bytes(bytes(range(200)))
    _, tr = K.permute(s, trace=True)
    return tr


def test_cell_index_matches_bits_layout():
    s = K.new_state()
    s[3, 1] = 1 << 7
    flat = bits_of(s)
    assert flat[anim.cell_index(3, 1, 7)] == 1
    assert flat.sum() == 1
    assert anim.XS[anim.cell_index(3, 1, 7)] == 3
    assert anim.YS[anim.cell_index(3, 1, 7)] == 1
    assert anim.ZS[anim.cell_index(3, 1, 7)] == 7


@pytest.mark.parametrize("step", K.STEP_NAMES)
def test_frame_endpoints_match_static_states(trace, step):
    snap = trace[trace.index_of(2, step)]
    prev = trace[snap.index - 1]
    pb, cb = bits_of(prev.state), bits_of(snap.state)
    f0 = anim.build_frame(step, pb, cb, 0.0, "raw", theta_c=snap.theta_c, theta_d=snap.theta_d)
    f1 = anim.build_frame(step, pb, cb, 1.0, "raw", theta_c=snap.theta_c, theta_d=snap.theta_d)
    col0, sc0 = anim.static_colors(pb, pb, "raw")
    col1, sc1 = anim.static_colors(cb, pb, "raw")
    assert np.allclose(f0.col, col0) and np.allclose(f0.scale, sc0)
    assert np.allclose(f1.col, col1) and np.allclose(f1.scale, sc1)
    assert np.array_equal(f0.pos, anim.BASE_POS) and np.array_equal(f1.pos, anim.BASE_POS)


def test_rho_animation_moves_cells_along_z_only(trace):
    snap = trace[trace.index_of(0, "rho")]
    prev = trace[snap.index - 1]
    pb, cb = bits_of(prev.state), bits_of(snap.state)
    f = anim.build_frame("rho", pb, cb, 0.5, "raw")
    assert np.allclose(f.pos[:, :2], anim.BASE_POS[:, :2])
    moved = np.abs(f.pos[:, 2] - anim.BASE_POS[:, 2])
    lane00 = anim.cell_index(0, 0, np.arange(64))
    assert np.allclose(moved[lane00], 0.0)  # r[0,0] = 0
    # a cell with offset r moves by r*smoothstep(0.5) = r/2 (mod 64)
    r = int(K.RHO_OFFSETS[1, 0])
    i = anim.cell_index(1, 0, 0)
    assert abs((anim.BASE_POS[i, 2] - f.pos[i, 2]) - r * 0.5) < 1e-4
    # the bit travels with its cell: colour at t<0.8 equals previous value's colour
    col_prev, _ = anim.static_colors(pb, pb, "raw")
    assert np.allclose(f.col, col_prev)


def test_pi_animation_targets(trace):
    snap = trace[trace.index_of(0, "pi")]
    prev = trace[snap.index - 1]
    pb, cb = bits_of(prev.state), bits_of(snap.state)
    f = anim.build_frame("pi", pb, cb, 0.999999, "raw")
    for x in range(5):
        for y in range(5):
            i = anim.cell_index(x, y, 3)
            tx, ty = K.pi_target(x, y)
            assert abs(f.pos[i, 0] - (tx - 2.0)) < 1e-3 and abs(f.pos[i, 1] - (ty - 2.0)) < 1e-3
            assert f.pos[i, 2] == anim.BASE_POS[i, 2]
    assert len(f.overlay_lines) > 0  # thin arrows


def test_theta_frame_has_sheets_and_lines(trace):
    snap = trace[trace.index_of(0, "theta")]
    prev = trace[snap.index - 1]
    pb, cb = bits_of(prev.state), bits_of(snap.state)
    f = anim.build_frame("theta", pb, cb, 0.5, "raw", theta_c=snap.theta_c, theta_d=snap.theta_d)
    assert f.extra_pos is not None and len(f.extra_pos) == 640
    assert f.instance_data().shape == (2240, 8)
    # C sheet cell (x, z) is on iff parity C[x] bit z
    cb_ = K.lanes_to_bits(snap.theta_c).reshape(5, 64)
    on = f.extra_scale[:320] > 0.5
    assert np.array_equal(on.reshape(5, 64), cb_.astype(bool))


def test_changed_mode_flags_exactly_the_flipped_bits(trace):
    snap = trace[trace.index_of(1, "chi")]
    prev = trace[snap.index - 1]
    pb, cb = bits_of(prev.state), bits_of(snap.state)
    col, _ = anim.static_colors(cb, pb, "changed")
    flipped = pb != cb
    is_red = np.all(np.isclose(col, anim.COL_CHANGED_TO_ONE), axis=1)
    is_blue = np.all(np.isclose(col, anim.COL_CHANGED_TO_ZERO), axis=1)
    assert np.array_equal(is_red | is_blue, flipped)
    assert np.array_equal(is_red, flipped & cb.astype(bool))


def test_structures():
    assert anim.structure_cells("row", 1, 2, 3).sum() == 5
    assert anim.structure_cells("column", 1, 2, 3).sum() == 5
    assert anim.structure_cells("lane", 1, 2, 3).sum() == 64
    assert anim.structure_cells("slice", 1, 2, 3).sum() == 25
    assert anim.structure_cells("plane", 1, 2, 3).sum() == 320
    assert anim.structure_cells("sheet", 1, 2, 3).sum() == 320
    lo, hi = anim.structure_bounds("lane", 1, 2, 3)
    assert np.allclose(lo, [-1.5, -0.5, -32.0]) and np.allclose(hi, [-0.5, 0.5, 32.0])


def test_camera_project_and_ray_agree():
    cam = OrbitCamera()
    for ortho in (False, True):
        cam.ortho = ortho
        mvp = cam.mvp(1.5)
        pt = np.array([[1.0, -0.5, 3.0]])
        sx, sy, w = cam.project(mvp, pt, 800, 600)[0]
        o, d = cam.ray(mvp, sx, sy, 800, 600)
        # the point must lie on the ray
        t = np.dot(pt[0] - o, d)
        assert np.linalg.norm(o + d * t - pt[0]) < 1e-3, ortho


def test_camera_presets_and_interaction():
    cam = OrbitCamera()
    cam.preset("slice-on")
    assert cam.ortho and cam.pitch == 0.0
    e = cam.eye()
    assert abs(e[0]) < 1e-9 and abs(e[1]) < 1e-9 and e[2] > 0  # camera on +Z, looking at z = 0 slice
    cam.preset("lane-on")
    assert cam.eye()[0] > 0
    d0 = cam.distance
    cam.zoom(1)
    assert cam.distance < d0
    cam.orbit(10, 0)
    cam.pan(5, 5, 600)
    assert np.isfinite(cam.mvp(1.0)).all()


def test_session_navigation():
    from keccakviz.ui.session import Params, Session

    s = Session(Params(message=b"x" * 300, num_rounds=3, show_load=False))
    assert s.run.num_perm_calls == 3
    assert s.num_snapshots == 16
    assert s.snap_index == 0
    assert s.step_forward() and s.snap_index == 1
    assert s.round_forward() and s.snap_index == 5
    assert s.round_forward() and s.snap_index == 10
    assert s.round_back() and s.snap_index == 5
    assert s.go_end() and s.snap_index == 15
    assert s.step_forward() and (s.perm_index, s.snap_index) == (1, 0)
    assert s.step_back() and (s.perm_index, s.snap_index) == (0, 15)
    assert s.go_start() and s.snap_index == 0
    assert not s.step_back()
    s.set_params(num_rounds=2)
    assert s.num_snapshots == 11
    s.set_params(num_rounds=24)
    s.apply_variant("SHAKE128")
    assert s.params.rate_bytes == 168 and s.params.domain_byte == 0x1F
    assert s.run.output == __import__("hashlib").shake_128(b"x" * 300).digest(32)
    av = s.avalanche()
    assert av.hamming(0) == 1
    d = s.diffusion()
    assert d.first_step[0, 0, 0] == 0
    assert "permutation 1/2" in s.position_text()


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
    return app


def test_scrubber_zones_hand_over_between_steps(qapp):
    from keccakviz.ui.session import Params, Session
    from keccakviz.ui.transport import Transport

    s = Session(Params(num_rounds=2, show_load=False))
    t = Transport(s)
    got = []
    t.scrubbed.connect(got.append)
    s.set_position(snap_index=3)
    # drag from the middle towards the right-hand zone
    t.step_slider.setValue(400)
    t._scrub_start()
    t.step_slider.setValue(500)  # middle of the scrub range (130 .. 870) -> t = 0.5
    assert abs(got[-1] - 0.5) < 0.01 and s.snap_index == 3
    t.step_slider.setValue(900)  # upper dead zone: sticks at the finished state
    assert got[-1] == 1.0 and t.step_slider.value() == 870 and s.snap_index == 3
    t.step_slider.setValue(980)  # into the end zone: next step, held at t = 0
    assert s.snap_index == 4 and got[-1] == 0.0 and t.step_slider.value() == 0 and t.step_slider.locked
    t.step_slider.setValue(995)  # still held: no further hand-over until release
    assert s.snap_index == 4
    t._scrub_end()
    assert not t.step_slider.locked
    # drag left into the start zone: previous step, shown complete
    t._scrub_start()
    t.step_slider.setValue(400)
    t.step_slider.setValue(10)
    assert s.snap_index == 3 and got[-1] == 1.0 and t.step_slider.value() == 1000
    t._scrub_end()
    # animator progress updates map into the scrub range and are ignored while dragging
    t.set_progress(0.5)
    assert t.step_slider.value() == 500
    t.set_progress(0.0)
    assert t.step_slider.value() == 130


def test_autoplay_stops_at_end_of_permutation(qapp):
    from keccakviz.ui.session import Params, Session
    from keccakviz.ui.transport import Transport

    s = Session(Params(message=b"x" * 300, num_rounds=1, show_load=False))
    t = Transport(s)
    s.set_position(snap_index=4)
    t.toggle_play()
    assert t.playing
    t._on_play_tick()  # 4 -> 5 (last snapshot of this call)
    assert s.snap_index == 5 and t.playing
    t._on_play_tick()  # at the end: stop, do not run into permutation 2
    assert not t.playing and (s.perm_index, s.snap_index) == (0, 5)


def test_styles_and_ghosting(trace):
    snap = trace[trace.index_of(0, "rho")]
    prev = trace[snap.index - 1]
    pb, cb = bits_of(prev.state), bits_of(snap.state)
    for style in anim.STYLES:
        col, sc = anim.static_colors(cb, pb, "raw", style=style)
        assert col.shape == (1600, 3) and sc.shape == (1600,)
    col, sc = anim.static_colors(cb, pb, "raw", style="ones")
    assert (sc[cb == 0] == 0).all() and (sc[cb == 1] > 0).all()
    f = anim.build_frame("rho", pb, cb, 0.5, "raw")
    assert f.alpha is None  # a lane slides as a whole: nothing crosses during rho
    ks = [anim.lane_crossing_strength(t / 20) for t in range(1, 20)]
    assert all(k.shape == (5, 5) for k in ks)
    assert any(k.max() == 1.0 for k in ks) and any(k.max() == 0.0 for k in ks)
    assert not anim.crossing_lanes(0.0).any() and not anim.crossing_lanes(1.0).any()
    f = anim.build_frame("pi", pb, cb, 0.5, "raw")
    assert f.alpha is not None and (f.alpha < 1).any() and (f.alpha == 1).any()
    arrows = anim.pi_arrows(1.0)
    assert len(arrows) == 24 * 2 * 3  # shaft + two head strokes per lane per face
    assert len(anim.pi_arrows(1.0, {(1, 0)})) == 2 * 3


def test_session_tracking_origin_and_positions():
    from keccakviz.ui.session import Params, Session

    s = Session(Params(num_rounds=3, show_load=False))
    s.set_position(snap_index=3)  # after theta, rho, pi of round 1
    cell_now = (2, 3, 10)
    origin = s.origin_of(cell_now)
    assert s.track(cell_now)
    assert [g.origins for g in s.tracked] == [[origin]]
    # the tracked bit sits exactly where we clicked at the current snapshot
    assert s.tracked_at()[0][1] == cell_now
    assert s.tracked_at(0)[0][1] == origin
    assert not s.track(cell_now)  # duplicate
    s.untrack(0)
    assert s.tracked == []
    # a whole structure through the cell, as one group
    s.set_structure("row")
    assert s.track_focus(cell_now)
    assert len(s.tracked) == 1 and len(s.tracked[0].origins) == 5 and not s.tracked[0].single
    now = {c for _ci, c in s.tracked_at()}
    assert now == {(x, cell_now[1], cell_now[2]) for x in range(5)}
    assert len(s.tracked_small()) == 5  # a row is a small group: boxes and trails
    s.set_structure("sheet")
    assert s.track_focus(cell_now) and len(s.tracked[1].origins) == 320
    assert len(s.tracked_big()) == 320  # big groups are tinted instead
    s.set_structure(None)
    assert s.track_focus((0, 0, 0)) and s.tracked[2].single
    assert len(s.tracked_small()) == 6


def test_loading_phase_in_session_and_dyes():
    from keccakviz.ui.session import Params, Session

    s = Session(Params(message=b"x" * 300, num_rounds=2, word_bits=32, words_per_cycle=4))
    n_words = 136 * 8 // 32
    n_load = 2 + -(-n_words // 4)
    assert s.n_load == n_load and s.num_snapshots == n_load + 11
    assert s.snapshot.step == "load" and "seed" in s.position_text()
    s.set_position(snap_index=n_load)
    assert s.snapshot.step == "initial" and s.core_index(s.snap_index) == 0
    assert s.round_forward() and s.snap_index == n_load + 5
    assert s.round_back() and s.snap_index == n_load
    assert s.round_back() and s.snap_index == 0
    assert s.trace[n_load + 5].round_constant == s.perm.trace[5].round_constant
    # squeeze calls of SHAKE have no loading phase
    s.set_params(show_load=False)
    assert s.n_load == 0
    # dyes: one bit -> spreads, conserved, homogeneous at the end
    s.set_params(show_load=True)
    s.set_position(snap_index=0)
    assert s.add_dye((0, 0, 0))
    assert s.color_mode == "dye" and len(s.dyes) == 1
    arr = s.dye_array()
    assert arr.shape == (s.num_snapshots, 1600, 1)
    assert s.dye_spread(0) == 1600.0 and s.dye_spread(s.num_snapshots - 1) < 10
    s.set_params(num_rounds=24)
    assert abs(s.dye_spread(s.num_snapshots - 1) - 1.0) < 0.05  # homogeneous after 24 rounds
    rgb, k = s.dye_render(0)
    assert rgb.shape == (1600, 3) and k.max() == 1.0 and (k > 0).sum() == 1
    # grouped (word) rendering: one dyed bit must still give full strength to its word
    from keccakviz.ui.glview import CubeView

    tbl = CubeView._word_index_table(64)
    mix, k = s.dye_render_groups(0, tbl)
    assert mix.shape == (25, 3) and k.max() == 1.0 and (k > 0).sum() == 1
    assert np.allclose(mix[k.argmax()], [1.0, 59 / 255, 59 / 255], atol=1e-3)
    s.set_structure("row")
    assert s.add_dye((1, 2, 3), "#00ff00") and len(s.dyes[1].origins) == 5
    s.remove_dye(0)
    assert len(s.dyes) == 1 and s.dyes[0].color == "#00ff00"
    # pulled-out regions offset positions of a region, not of bits
    s.set_structure(None)
    assert s.pull_out("slice", (0, 0, 5))
    off = s.pull_offsets()
    assert off.shape == (1600, 3) and (off[:, 1] > 0).sum() == 25
    s.push_back()
    assert s.pull_offsets() is None


def test_load_frame_animation_and_offsets(trace):
    from keccakviz.core import sponge as S

    run = S.sponge(b"x" * 300, S.VARIANTS["SHA3-256"])
    blk = run.absorb_blocks[1]
    ft = S.full_trace(blk.perm, blk, 64, 1)
    snap = ft[3]  # second bus cycle
    prev = ft[2]
    pb, cb = bits_of(prev.state), bits_of(snap.state)
    cells = np.array([anim.cell_index(*K.bit_coords(i)) for i in snap.info["cells"]])
    f = anim.build_frame("load", pb, cb, 0.5, "raw", load_cells=cells)
    moved = np.abs(f.pos - anim.BASE_POS).sum(axis=1) > 1e-6
    assert moved.sum() == 64 and set(np.nonzero(moved)[0]) == set(cells)
    f1 = anim.build_frame("load", pb, cb, 1.0, "raw", load_cells=cells)
    assert np.array_equal(f1.pos, anim.BASE_POS)
    # offsets lift a region; during rho a cell blends between its old and new region offset
    off = np.zeros((1600, 3), dtype=np.float32)
    off[anim.structure_cells("slice", 0, 0, 5)] = (0, 8, 0)
    snap = trace[trace.index_of(0, "chi")]
    pb, cb = bits_of(trace[snap.index - 1].state), bits_of(snap.state)
    f = anim.build_frame("chi", pb, cb, 0.3, "raw", offsets=off)
    lifted = f.pos[:, 1] - anim.BASE_POS[:, 1] > 7
    assert lifted.sum() == 25
    snap = trace[trace.index_of(0, "rho")]
    pb, cb = bits_of(trace[snap.index - 1].state), bits_of(snap.state)
    f = anim.build_frame("rho", pb, cb, 0.5, "raw", offsets=off)
    i = anim.cell_index(1, 0, 4)  # moves by r[1,0] = 1 into slice 5: half lifted at t = 0.5
    assert 3 < f.pos[i, 1] - anim.BASE_POS[i, 1] < 5
