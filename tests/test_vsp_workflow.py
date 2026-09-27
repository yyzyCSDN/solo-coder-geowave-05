import numpy as np
from dataclasses import replace
from geowave.model.traces import TraceHeader, Trace
from geowave.model.wavelet import ricker
from geowave.signal.statics import apply_static
from geowave.well.checkshot import Checkshot
from geowave.vsp.gather import VSPGather, VSPLevel, from_trace_set
from geowave.vsp.geometry import validate_depth_geometry
from geowave.vsp.first_break import pick_first_breaks, align_on_first_break
from geowave.vsp.separation import median_separation
from geowave.vsp.corridor import corridor_stack
from geowave.vsp.tie import tie_to_surface, checkshot_calibration
from geowave.vsp.pipeline import process_vsp

V = 2500.0
DEPTHS = np.arange(500.0, 1501.0, 25.0)
REFLECTORS = [(1700.0, 0.25), (2000.0, -0.2), (2400.0, 0.22)]
TWTS = [2 * z / V for z, _ in REFLECTORS]
DT = 0.001
N = 2001

def _place(w, n, t0):
    out = np.zeros(n)
    start = int(round((t0 - (len(w) - 1) / 2 * DT) / DT))
    i0 = max(0, start)
    i1 = min(n, start + len(w))
    if i1 > i0:
        out[i0:i1] = w[i0 - start:i1 - start]
    return out

def make_gather(polarity=1.0, noise=0.005, seed=7, order=None):
    rng = np.random.default_rng(seed)
    w = ricker(25.0, DT, 0.128).samples
    levels = []
    for i, z in enumerate(DEPTHS):
        tfb = z / V
        s = _place(w, N, tfb)
        for zr, r in REFLECTORS:
            if zr > z:
                s = s + r * _place(w, N, 2 * zr / V - tfb)
        s = polarity * s + noise * rng.standard_normal(N)
        h = TraceHeader(f'z{i}', 'shot0', f'g{i}', 0.0, 0.0, 0.0, 0.0, DT, 0.0)
        levels.append(VSPLevel(float(z), Trace(h, s)))
    if order is not None:
        levels = [levels[i] for i in order]
    return VSPGather(tuple(levels), 'synthetic', 'KB')

def make_surface():
    w = ricker(25.0, DT, 0.128).samples
    s = np.zeros(N)
    for (zr, r), twt in zip(REFLECTORS, TWTS):
        s = s + r * _place(w, N, twt)
    return Trace(TraceHeader('surf', 's0', 'well', 0.0, 0.0, 0.0, 0.0, DT, 0.0), s)

def peak_near(stack, twt, t0, tol=0.08):
    m = (twt >= t0 - tol) & (twt <= t0 + tol)
    k = int(np.argmax(np.abs(stack[m])))
    return float(twt[m][k]), float(stack[m][k])

def test_geometry_clean():
    qc = validate_depth_geometry(make_gather())
    assert qc.monotonic_with_order and not qc.duplicate_levels
    assert qc.issues == () and qc.level_count == len(DEPTHS)

def test_geometry_depth_order_and_datum_flagged():
    g = make_gather(order=np.random.default_rng(3).permutation(len(DEPTHS)))
    qc = validate_depth_geometry(g)
    assert not qc.monotonic_with_order
    assert 'depth-order' in {i.code for i in qc.issues}
    assert any('井深' in i.impact for i in qc.issues)

def test_geometry_duplicate_depth_flagged():
    g = make_gather()
    levels = list(g.levels)
    levels.insert(5, levels[5])
    qc = validate_depth_geometry(VSPGather(tuple(levels), 'dup', 'KB'))
    assert 'duplicate-depth' in {i.code for i in qc.issues}
    assert any(i.severity == 'critical' for i in qc.issues)

def test_from_trace_set():
    g = make_gather()
    from geowave.model.traces import TraceSet
    ds = TraceSet([lv.trace for lv in g.levels], 'ts')
    g2 = from_trace_set(ds, {lv.trace.header.trace_id: lv.depth_m for lv in g.levels})
    assert np.allclose(g2.depths_m, DEPTHS)

def test_first_break_picks_track_velocity():
    g = make_gather()
    al = pick_first_breaks(g)
    assert al.low_confidence_levels == ()
    assert np.allclose(al.picks_s, DEPTHS / V, atol=0.06)
    flat = align_on_first_break(g, al)
    m = flat.to_matrix()
    lo = int((al.reference_s - 0.06) / DT)
    hi = int((al.reference_s + 0.08) / DT)
    peak_idx = np.argmax(np.abs(m[:, lo:hi]), axis=1)
    peak_amp = np.abs(m[:, lo:hi]).max(axis=1)
    assert np.all(peak_amp > 0.5)
    assert np.std(peak_idx * DT) < 0.002

def test_separation_removes_downgoing():
    g = make_gather()
    al = pick_first_breaks(g)
    sep = median_separation(align_on_first_break(g, al), 5)
    k = int(round(al.reference_s / DT))
    down_energy = float(np.sum(sep.downgoing.to_matrix()[:, k - 3:k + 4] ** 2))
    up_energy = float(np.sum(sep.upgoing.to_matrix()[:, k - 3:k + 4] ** 2))
    assert down_energy > 5 * up_energy
    assert sep.upgoing_energy < sep.downgoing_energy

def test_corridor_stack_recovers_reflectors():
    g = make_gather()
    al = pick_first_breaks(g)
    sep = median_separation(align_on_first_break(g, al), 5)
    cs = corridor_stack(sep.upgoing, al, (0.0, 1.0))
    for twt, (_, r) in zip(TWTS, REFLECTORS):
        t_peak, amp = peak_near(cs.stack, cs.twt_s, twt)
        assert abs(t_peak - twt) < 0.08
        assert np.sign(amp) == np.sign(r)
    assert cs.levels_used == len(DEPTHS)

def test_pipeline_clean_has_no_impacts():
    res = process_vsp(make_gather(), corridor_s=(0.0, 1.0))
    assert res.grade == 'A'
    assert res.impacts == ()

def test_pipeline_surface_tie():
    res = process_vsp(make_gather(), surface=make_surface(), corridor_s=(0.0, 1.0))
    tie = res.surface_tie
    assert tie is not None and not tie.polarity_flipped
    assert abs(tie.shift_s) < 0.08
    assert tie.correlation > 0.5

def test_pipeline_polarity_doubt_written_to_impacts():
    res = process_vsp(make_gather(polarity=-1.0), surface=make_surface(), corridor_s=(0.0, 1.0))
    assert res.surface_tie.polarity_flipped
    assert any('极性' in s for s in res.impacts)
    tie2 = tie_to_surface(res.corridor, make_surface())
    assert not tie2.polarity_flipped and tie2.correlation > 0.5

def test_pipeline_shuffled_depths_reordered():
    g = make_gather(order=np.random.default_rng(11).permutation(len(DEPTHS)))
    res = process_vsp(g, corridor_s=(0.0, 1.0))
    assert any('井深' in s for s in res.impacts)
    assert res.grade in ('B', 'C', 'D')
    for twt in TWTS:
        t_peak, _ = peak_near(res.corridor.stack, res.corridor.twt_s, twt)
        assert abs(t_peak - twt) < 0.08

def test_pipeline_mixed_time_datum_unified():
    g = make_gather()
    levels = []
    for i, lv in enumerate(g.levels):
        if 10 <= i < 20:
            tr = apply_static(lv.trace, -0.05)
            tr = Trace(replace(tr.header, start_s=0.05), tr.samples, tr.tags)
            lv = VSPLevel(lv.depth_m, tr)
        levels.append(lv)
    res = process_vsp(VSPGather(tuple(levels), 'mixed', 'KB'), corridor_s=(0.0, 1.0))
    assert 'mixed-time-datum' in {i.code for i in res.geometry.issues}
    assert any('采样基准' in s for s in res.impacts)
    for twt in TWTS:
        t_peak, _ = peak_near(res.corridor.stack, res.corridor.twt_s, twt)
        assert abs(t_peak - twt) < 0.08

def test_pipeline_checkshot_datum_offset():
    shots = [Checkshot(float(z), 2 * float(z) / V) for z in DEPTHS[::10]]
    res = process_vsp(make_gather(), checkshots=shots, corridor_s=(0.0, 1.0))
    assert abs(res.checkshot_tie.mean_s) < 0.06
    assert any('井震标定' in s for s in res.impacts)
    relaxed = process_vsp(make_gather(), checkshots=shots, corridor_s=(0.0, 1.0), checkshot_tol_s=0.06)
    assert not any('井震标定' in s for s in relaxed.impacts)
    shifted = [Checkshot(c.depth_m, c.two_way_time_s + 0.02) for c in shots]
    tie = checkshot_calibration(DEPTHS, DEPTHS / V, shifted)
    assert abs(tie.mean_s + 0.01) < 1e-9

def test_pipeline_duplicate_levels_degraded_but_processed():
    g = make_gather()
    levels = list(g.levels)
    levels.insert(5, levels[5])
    res = process_vsp(VSPGather(tuple(levels), 'dup', 'KB'), corridor_s=(0.0, 1.0))
    assert res.grade == 'D'
    assert any('重复' in i.detail for i in res.geometry.issues)
    for twt in TWTS:
        t_peak, _ = peak_near(res.corridor.stack, res.corridor.twt_s, twt)
        assert abs(t_peak - twt) < 0.08
