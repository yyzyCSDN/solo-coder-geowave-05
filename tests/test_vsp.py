import numpy as np
import pytest
from geowave.vsp import (VSPGather, run_vsp_pipeline, check_geometry,
                         regularize_levels, pick_first_breaks,
                         median_separate, fk_separate,
                         vsp_checkshots, Issue)
from geowave.well.depth_time import build_depth_time

REFLECTORS = (1.4,)

def _ricker(t, t0, f=30.0):
    u = np.pi * f * (t - t0)
    return (1.0 - 2.0 * u * u) * np.exp(-u * u)

def _synthetic_vsp(n_levels=66, z0=700.0, dz=10.0, dt=0.002, n=1001,
                   velocity=2000.0, interfaces_twt=REFLECTORS,
                   polarity_flip=(), reverse_rows=False, noise=0.0,
                   seed=0, datum_verified=True, depth_verified=True,
                   polarity_verified=True, source_offset_m=0.0):
    """Build a zero-offset VSP: downgoing direct wave + upgoing reflectors.

    Direct arrival at receiver z is tb = z/v. The reflection from an
    interface at two-way time tau (depth H = v*tau/2) travels H down plus
    H-z back up: t = (2H-z)/v = tau - tb. Every receiver above the
    interface sees it; the event moves out opposite to the first break.
    Receivers (700-1300 m) stay above the shallowest interface (1400 m).
    """
    rng = np.random.default_rng(seed)
    z = z0 + np.arange(n_levels) * dz
    t = np.arange(n) * dt
    data = np.zeros((n_levels, n))
    for i, zi in enumerate(z):
        tb = np.hypot(zi, source_offset_m) / velocity
        x = _ricker(t, tb)
        for tau in interfaces_twt:
            x += 0.8 * _ricker(t, tau - tb)
        x += noise * rng.standard_normal(n)
        if i in polarity_flip:
            x = -x
        data[i] = x
    ids = [f'r{i}' for i in range(n_levels)]
    if reverse_rows:
        order = np.arange(n_levels)[::-1]
        data = data[order]
        z = z[order]
        ids = [ids[i] for i in order]
    return VSPGather(z, data, dt, receiver_ids=ids, name='synthetic-vsp',
                     source_offset_m=source_offset_m,
                     datum_verified=datum_verified,
                     depth_verified=depth_verified,
                     polarity_verified=polarity_verified), t

def test_geometry_depth_reorder_and_irregular():
    vsp, _ = _synthetic_vsp(reverse_rows=True)
    qc = check_geometry(vsp)
    assert np.all(np.diff(qc.depths_m) > 0)
    codes = {i.code for i in qc.issues}
    assert 'depth-order' in codes
    # raw rows got put back into depth order
    assert np.allclose(qc.raw_data[0], vsp.data[-1])

    vsp2, _ = _synthetic_vsp()
    z = vsp2.depths_m.copy()
    z[15] += 4.0  # 4 m jump vs 10 m spacing
    vsp2.depths_m = z
    qc2 = check_geometry(vsp2)
    assert 'irregular-spacing' in {i.code for i in qc2.issues}
    reg, panel = regularize_levels(qc2)
    assert np.allclose(np.diff(reg), reg[1] - reg[0])

def test_geometry_unverified_datum_and_depth_emit_impact():
    vsp, _ = _synthetic_vsp(datum_verified=False, depth_verified=False)
    res = run_vsp_pipeline(vsp)
    codes = {i.code for i in res.issues}
    assert {'datum-unverified', 'depth-unverified'} <= codes
    for i in res.issues:
        assert i.impact.strip()
    report = res.impact_report()
    assert 'impact:' in report and 'datum' in report.lower()

def test_first_break_picks_and_interval_velocity():
    vsp, _ = _synthetic_vsp()
    qc = check_geometry(vsp)
    depths, panel = regularize_levels(qc)
    fb = pick_first_breaks(depths, panel, vsp.dt_s)
    expected = depths / 2000.0
    assert np.max(np.abs(fb.times_s - expected)) < 0.005
    assert abs(fb.fit.apparent_velocity_mps - 2000.0) < 30.0
    assert abs(fb.fit.intercept_s) < 0.002
    assert np.all((fb.fit.interval_velocities_mps > 1900) &
                  (fb.fit.interval_velocities_mps < 2100))

def test_polarity_detection_and_fix():
    vsp, _ = _synthetic_vsp(polarity_flip=(7, 22))
    res = run_vsp_pipeline(vsp)
    assert res.polarity_flipped == [7, 22]
    assert 'polarity-reversed' in {i.code for i in res.issues}
    vsp2, _ = _synthetic_vsp(polarity_flip=(7, 22))
    res2 = run_vsp_pipeline(vsp2, fix_polarity=False)
    assert 'polarity-unfixed' in {i.code for i in res2.issues}

def test_separation_median_keeps_upgoing_reflections():
    vsp, _ = _synthetic_vsp()
    qc = check_geometry(vsp)
    depths, panel = regularize_levels(qc)
    fb = pick_first_breaks(depths, panel, vsp.dt_s)
    from geowave.vsp import align_first_breaks
    flat, target = align_first_breaks(panel, fb.times_s, vsp.dt_s)
    wf = median_separate(flat, 3)
    # Direct wave is flattened at the shallowest pick (~0.35 s). The 1.4 s
    # reflector maps to tau - 2tb + target ~ 1.05-1.4 s, well separated.
    jd = int(target / vsp.dt_s)
    down_direct = np.sum(np.abs(wf.downgoing[:, jd - 12:jd + 12]))
    jlo, jhi = int(1.1 / vsp.dt_s), int(1.35 / vsp.dt_s)
    down_far = np.sum(np.abs(wf.downgoing[:, jlo:jhi]))
    assert down_direct > 5 * down_far
    up = np.sum(np.abs(wf.upgoing[:, jlo:jhi]))
    assert up > 10 * down_far

def test_corridor_stack_event_at_reflector_twt():
    vsp, _ = _synthetic_vsp()
    res = run_vsp_pipeline(vsp)
    c = res.corridor
    for tau in REFLECTORS:
        j = int(np.argmin(np.abs(c.times_s - tau)))
        seg = c.narrow[j - 25:j + 25]
        peak = j - 25 + int(np.argmax(np.abs(seg)))
        assert abs(c.times_s[peak] - tau) < 0.02
    # levels with tb in [tau/2, tau/2+narrow] contribute -> real fold
    assert c.fold_narrow[np.argmin(np.abs(c.times_s - 1.4))] >= 3
    # past the deepest mapped first-break line there is no corridor coverage
    assert c.fold_narrow[-1] == 0

def test_corridor_ties_to_surface_seismic_and_synthetic():
    vsp, t = _synthetic_vsp()
    # surface seismic shares the same wavelet/time base with a +4 ms static
    seismic = sum(_ricker(t, tau + 0.004) for tau in REFLECTORS)
    synthetic = sum(_ricker(t, tau) for tau in REFLECTORS)
    res = run_vsp_pipeline(vsp, target_times_s=t,
                           surface_seismic=seismic, synthetic=synthetic,
                           max_tie_shift_s=0.05)
    assert res.surface_tie is not None and res.synthetic_tie is not None
    assert abs(res.surface_tie.shift_s - (-0.004)) < 0.0021
    assert abs(res.synthetic_tie.shift_s) < 0.0021
    assert res.synthetic_tie.correlation > 0.6
    assert res.surface_tie.correlation > 0.6

def test_checkshots_are_two_way_and_feed_depth_time_model():
    vsp, _ = _synthetic_vsp()
    res = run_vsp_pipeline(vsp)
    cs = res.checkshots()
    assert len(cs) == res.depths_m.size
    model = build_depth_time(cs)
    assert abs(model.time_at_depth(700.0) - 0.7) < 0.005
    assert abs(model.time_at_depth(1000.0) - 1.0) < 0.005

def test_fk_separation_path_runs_and_aligns():
    vsp, _ = _synthetic_vsp()
    res = run_vsp_pipeline(vsp, method='fk')
    assert res.wavefields.method == 'fk-fan'
    j = int(np.argmin(np.abs(res.corridor.times_s - 1.4)))
    seg = res.corridor.narrow[j - 25:j + 25]
    peak = j - 25 + int(np.argmax(np.abs(seg)))
    assert abs(res.corridor.times_s[peak] - 1.4) < 0.04

def test_offset_vsp_is_flagged_and_corrected():
    # shallow short array, one deep reflector, so direct and reflected
    # arrivals stay separated for the picker
    vsp, t = _synthetic_vsp(n_levels=21, z0=150.0, dz=10.0,
                            interfaces_twt=(1.4, 1.5),
                            source_offset_m=250.0)
    res = run_vsp_pipeline(vsp)
    codes = {i.code for i in res.issues}
    assert 'offset-vsp' in codes and 'offset-correction' in codes
    # corrected vertical first break at shallowest level ~ z/v, not sqrt/V
    assert abs(res.corrected_first_breaks_s[0] -
               vsp.depths_m[0] / 2000.0) < 0.008

def test_noise_robust_pipeline():
    vsp, _ = _synthetic_vsp(noise=0.05)
    res = run_vsp_pipeline(vsp)
    assert abs(res.first_breaks.fit.apparent_velocity_mps - 2000.0) < 60.0
    c = res.corridor
    j = int(np.argmin(np.abs(c.times_s - 1.4)))
    lo, hi = max(0, j - 25), min(c.times_s.size, j + 25)
    peak = lo + int(np.argmax(np.abs(c.narrow[lo:hi])))
    assert abs(c.times_s[peak] - 1.4) < 0.03

def test_issue_dataclass_validates_severity():
    with pytest.raises(ValueError):
        Issue('x', 'catastrophic', 'm', 'i')

def test_empty_or_short_search_raises():
    vsp, _ = _synthetic_vsp()
    with pytest.raises(ValueError):
        run_vsp_pipeline(vsp, search_s=(0.0, 0.001))
