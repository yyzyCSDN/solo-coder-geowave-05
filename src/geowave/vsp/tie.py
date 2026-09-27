from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate
from geowave.model.traces import Trace
from geowave.well.checkshot import Checkshot
from geowave.vsp.corridor import CorridorStackResult

@dataclass(frozen=True)
class SurfaceTie:
    shift_s: float
    correlation: float
    scale: float
    polarity_flipped: bool
    polarity_margin: float

@dataclass(frozen=True)
class CheckshotTie:
    residuals_s: np.ndarray
    mean_s: float
    rms_s: float
    max_abs_s: float

def tie_to_surface(corridor: CorridorStackResult, surface: Trace, max_shift_s: float=0.2) -> SurfaceTie:
    dt = float(corridor.twt_s[1] - corridor.twt_s[0]) if corridor.twt_s.size > 1 else surface.header.dt_s
    if abs(surface.header.dt_s - dt) > 1e-9:
        raise ValueError('sampling mismatch')
    a = corridor.stack
    b = surface.samples
    n = min(a.size, b.size)
    a0 = a[:n] - np.mean(a[:n])
    b0 = b[:n] - np.mean(b[:n])
    c = correlate(a0, b0, mode='full')
    lags = np.arange(-n + 1, n)
    mask = np.abs(lags) <= int(round(max_shift_s / dt))
    cw = c[mask]
    lw = lags[mask]
    norm = float(np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-12
    i_pos = int(np.argmax(cw))
    i_neg = int(np.argmin(cw))
    corr_pos = float(cw[i_pos] / norm)
    corr_neg = float(-cw[i_neg] / norm)
    flipped = corr_neg > corr_pos
    lag = lw[i_neg] if flipped else lw[i_pos]
    idx = np.arange(n) - lag
    shifted = np.interp(idx, np.arange(n), a0, left=0, right=0)
    if flipped:
        shifted = -shifted
    den = float(np.dot(shifted, shifted))
    scale = float(np.dot(b0, shifted) / den) if den else 1.0
    return SurfaceTie(float(lag * dt), corr_neg if flipped else corr_pos, scale, flipped, corr_neg - corr_pos)

def checkshot_calibration(depths_m: np.ndarray, picks_s: np.ndarray, checkshots: list[Checkshot]) -> CheckshotTie:
    if not checkshots:
        raise ValueError('empty checkshot list')
    cs = sorted(checkshots, key=lambda c: c.depth_m)
    d = np.asarray([c.depth_m for c in cs], float)
    one_way_s = np.asarray([c.two_way_time_s for c in cs], float) / 2
    order = np.argsort(depths_m)
    pick_at = np.interp(d, np.asarray(depths_m, float)[order], np.asarray(picks_s, float)[order])
    resid = pick_at - one_way_s
    return CheckshotTie(resid, float(np.mean(resid)), float(np.sqrt(np.mean(resid ** 2))), float(np.max(np.abs(resid))))
