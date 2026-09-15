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
    assert len(f.overlay_lines) > 0  # arrows


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

    s = Session(Params(message=b"x" * 300, num_rounds=3))
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

    s = Session(Params(num_rounds=2))
    t = Transport(s)
    got = []
    t.scrubbed.connect(got.append)
    s.set_position(snap_index=3)
    # drag from the middle towards the right-hand zone
    t.step_slider.setValue(500)
    t._scrub_start()
    t.step_slider.setValue(600)
    assert got[-1] == 0.6 and s.snap_index == 3
    t.step_slider.setValue(980)  # into the zone: next step, restarted at t = 0
    assert s.snap_index == 4 and got[-1] == 0.0 and t.step_slider.value() == 0
    t.step_slider.setValue(995)  # still held in the zone: no further hand-over until release
    assert s.snap_index == 4
    t._scrub_end()
    # drag left into the start zone: previous step, shown complete
    t._scrub_start()
    t.step_slider.setValue(400)
    t.step_slider.setValue(10)
    assert s.snap_index == 3 and got[-1] == 1.0 and t.step_slider.value() == 1000
    t._scrub_end()
    # view progress updates are ignored while dragging, applied otherwise
    t.set_progress(0.25)
    assert t.step_slider.value() == 250


def test_session_tracking_origin_and_positions():
    from keccakviz.ui.session import Params, Session

    s = Session(Params(num_rounds=3))
    s.set_position(snap_index=3)  # after theta, rho, pi of round 1
    cell_now = (2, 3, 10)
    origin = s.origin_of(cell_now)
    assert s.track(cell_now)
    assert s.tracked == [origin]
    # the tracked bit sits exactly where we clicked at the current snapshot
    assert s.tracked_at()[0][1] == cell_now
    assert s.tracked_at(0)[0][1] == origin
    assert not s.track(cell_now)  # duplicate
    s.untrack(origin)
    assert s.tracked == []
