from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.model.traces import Trace
from geowave.signal.statics import apply_static
from geowave.velocity.first_break import FirstBreakPick, aic_pick
from geowave.vsp.gather import VSPGather

@dataclass(frozen=True)
class FirstBreakAlignment:
    picks_s: np.ndarray
    confidence: np.ndarray
    reference_s: float
    trend_s: np.ndarray
    residual_s: np.ndarray
    low_confidence_levels: tuple[int, ...]

def _gated_sta_lta(trace: Trace, sta_s: float, lta_s: float, threshold: float, energy_factor: float) -> FirstBreakPick | None:
    dt = trace.header.dt_s
    sta = max(1, int(sta_s / dt))
    lta = max(sta + 1, int(lta_s / dt))
    x = trace.samples
    if x.size <= lta:
        return None
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    i = np.arange(lta, x.size)
    short = (cs[i + 1] - cs[i + 1 - sta]) / sta
    long = (cs[i + 1 - sta] - cs[i + 1 - lta]) / max(1, lta - sta)
    ratio = short / (long + 1e-12)
    floor = energy_factor * float(np.median(short))
    valid = (ratio > threshold) & (short > floor)
    if not valid.any():
        return None
    k = int(i[np.argmax(valid)])
    confidence = min(1.0, (float(ratio[np.argmax(valid)]) - threshold) / max(threshold, 1e-09))
    return FirstBreakPick(trace.header.trace_id, trace.header.start_s + k * dt, confidence, 'gated-sta-lta')

def _robust_line(x: np.ndarray, y: np.ndarray, tol_s: float, iterations: int=200) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n < 3:
        coef = np.polyfit(x, y, 1) if n == 2 else np.array([0.0, float(y[0])])
        return (coef, np.ones(n, bool))
    rng = np.random.default_rng(0)
    best_in = np.ones(n, bool)
    best_count = -1
    for _ in range(iterations):
        a, b = rng.choice(n, 2, replace=False)
        if x[a] == x[b]:
            continue
        coef = np.polyfit([x[a], x[b]], [y[a], y[b]], 1)
        inl = np.abs(y - np.polyval(coef, x)) <= tol_s
        if inl.sum() > best_count:
            best_count = int(inl.sum())
            best_in = inl
    coef = np.polyfit(x[best_in], y[best_in], 1) if best_count >= 2 else np.polyfit(x, y, 1)
    return (coef, best_in)

def pick_first_breaks(gather: VSPGather, sta_s: float=0.01, lta_s: float=0.08, threshold: float=2.5, refine_window_s: float=0.06, residual_factor: float=3.0, trend_tol_s: float=0.01, repick_window_s: float=0.06, energy_factor: float=10.0) -> FirstBreakAlignment:
    n = len(gather.levels)
    picks = np.zeros(n)
    confidence = np.zeros(n)
    for i, lv in enumerate(gather.levels):
        pick = _gated_sta_lta(lv.trace, sta_s, lta_s, threshold, energy_factor)
        if pick is not None:
            half = refine_window_s / 2
            refined = aic_pick(lv.trace, pick.time_s - half, pick.time_s + half)
            if refined is not None:
                pick = FirstBreakPick(pick.trace_id, refined.time_s, max(pick.confidence, refined.confidence), 'gated-sta-lta+aic')
        if pick is None:
            pick = aic_pick(lv.trace)
        if pick is None:
            pick = FirstBreakPick(lv.trace.header.trace_id, lv.trace.header.start_s, 0.0, 'none')
        picks[i] = pick.time_s
        confidence[i] = pick.confidence
    depths = gather.depths_m
    if n >= 3:
        coef, inliers = _robust_line(depths, picks, trend_tol_s)
        for i in np.where(~inliers)[0]:
            pred = float(np.polyval(coef, depths[i]))
            half = repick_window_s / 2
            repick = aic_pick(gather.levels[i].trace, pred - half, pred + half)
            if repick is not None:
                picks[i] = repick.time_s
                confidence[i] = 0.5
        coef, _ = _robust_line(depths, picks, trend_tol_s)
        trend = np.polyval(coef, depths)
    else:
        trend = picks.copy()
    residual = picks - trend
    mad = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    tol = max(residual_factor * mad, 2 * gather.dt_s)
    low = tuple((int(i) for i in np.where((np.abs(residual - np.median(residual)) > tol) | (confidence < 0.2))[0]))
    return FirstBreakAlignment(picks, confidence, float(np.median(picks)), trend, residual, low)

def align_on_first_break(gather: VSPGather, alignment: FirstBreakAlignment, reference_s: float | None=None) -> VSPGather:
    ref = alignment.reference_s if reference_s is None else reference_s
    traces = [apply_static(lv.trace, ref - float(p)) for lv, p in zip(gather.levels, alignment.picks_s)]
    return gather.with_traces(traces, gather.name + '-fbalign')
