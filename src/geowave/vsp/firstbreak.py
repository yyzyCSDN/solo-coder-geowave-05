from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DirectArrivalFit:
    """Linear fit t(z) = intercept_s + z / apparent_velocity_mps.

    For a zero-offset VSP the apparent velocity is the average P velocity;
    for an offset VSP it carries the horizontal slowness and is corrected
    with the straight-ray source offset.
    """
    intercept_s: float
    apparent_velocity_mps: float
    residuals_s: np.ndarray
    rms_residual_s: float
    interval_velocities_mps: np.ndarray

@dataclass(frozen=True)
class FirstBreaks:
    times_s: np.ndarray        # NaN where no confident pick exists
    confidence: np.ndarray
    fit: DirectArrivalFit
    corrected_times_s: np.ndarray

def _energy_onset(x: np.ndarray, i0: int, i1: int,
                  energy_window: int) -> int:
    """Earliest sample where short-term energy jumps out of pre-arrival noise.

    A global AIC is unusable on VSP traces because the quiet tail after the
    last reflection scores even lower than the first-break onset, so the
    onset has to be bounded by an energy trigger first.
    """
    if i1 - i0 < energy_window + 2:
        return -1
    kernel = np.ones(energy_window) / energy_window
    env = np.convolve(x * x, kernel, mode='same')
    # Baseline from the quietest part of the whole search interval rather
    # than its first samples: a shallow first break's leading side lobes can
    # otherwise fill a fixed-length pre-arrival window and hide the onset.
    bg = float(np.quantile(env[i0:i1], 0.10))
    bg_sd = float(np.std(env[env <= np.quantile(env[i0:i1], 0.25)]))
    peak = float(np.max(env[i0:i1]))
    # Threshold combines a statistical noise trigger and a relative floor;
    # on noise-free synthetic data bg ~ 0 and the floor alone must apply.
    thr = max(bg + 6.0 * bg_sd, 0.02 * peak, 20.0 * bg)
    det = i0
    while det < i1 and env[det] <= thr:
        det += 1
    return det if det < i1 else -1

def _aic_confidence(x: np.ndarray, det: int, ew: int) -> float:
    """Onset contrast (variance ratio) around the energy trigger, 0..1."""
    seg = x[max(0, det - 2 * ew):det + 2 * ew]
    if seg.size < 6:
        return 0.0
    k = min(2 * ew, seg.size - 2)
    pre = float(np.std(seg[:k])) + 1e-12
    post = float(np.std(seg[k:])) + 1e-12
    return min(1.0, max(0.0, (post / pre - 1.0) / 7.0))

def pick_first_breaks(depths_m: np.ndarray, data: np.ndarray, dt_s: float,
                      start_s: float = 0.0,
                      search_s: tuple[float, float] | None = None,
                      main_period_s: float = 0.035,
                      min_confidence: float = 0.1) -> FirstBreaks:
    """Gated main-peak first-break picks with sub-sample resolution.

    An energy trigger bounds the onset (a global AIC would lock onto the
    quiet tail of the record); the largest absolute sample within the first
    main wavelet period after the trigger is the first-break phase, refined
    to sub-sample accuracy by parabolic interpolation. Picking the same
    phase at every level avoids the integer-sample jitter that aliases the
    t-z slope when moveout per level is a fraction of a sample. Picks are
    fit by a straight line in t-z and off-line picks are replaced by the fit
    and flagged zero confidence.
    """
    n, ns = data.shape
    if search_s is None:
        i0, i1 = 1, ns - 2
    else:
        i0 = max(1, int((search_s[0] - start_s) / dt_s))
        i1 = min(ns - 2, int((search_s[1] - start_s) / dt_s))
    ew = max(2, int(0.01 / dt_s))
    period = max(3, int(main_period_s / dt_s))

    picks = np.full(n, np.nan)
    conf = np.zeros(n)
    for i in range(n):
        det = _energy_onset(data[i], i0, i1, ew)
        if det < 0:
            continue
        a = det
        b = min(i1 - 1, det + period)
        k = a + int(np.argmax(np.abs(data[i, a:b + 1])))
        y0, y1, y2 = (abs(data[i, k - 1]), abs(data[i, k]),
                      abs(data[i, k + 1]))
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        frac = float(np.clip(frac, -1.0, 1.0))
        picks[i] = start_s + (k + frac) * dt_s
        conf[i] = _aic_confidence(data[i], det, ew)

    z = np.asarray(depths_m, float)
    good = np.isfinite(picks) & (conf >= min_confidence)
    if good.sum() < 2:
        raise ValueError('fewer than two confident first-break picks; '
                         'widen the search window or supply manual picks')
    # Center both axes before the normal equations: moveout per level can be
    # a fraction of a sample while absolute times are ~0.1 s, and an
    # uncentred least-squares fit loses the slope to floating-point noise.
    zc = float(np.mean(z[good]))
    tc = float(np.mean(picks[good]))
    slope = float(np.sum((z[good] - zc) * (picks[good] - tc)) /
                  np.sum((z[good] - zc) ** 2))
    intercept = tc - slope * zc
    if not 1.0 / 8000.0 < slope < 1.0 / 1000.0:
        raise ValueError('fitted first-break slope implies velocity outside '
                         '1000-8000 m/s')
    a, b = slope, intercept
    vel = 1.0 / slope

    line = a * z + b
    weak = ~good
    picks_out = np.where(weak | (np.abs(picks - line) > 0.5 * main_period_s),
                         line, picks)
    conf_out = np.where(weak, 0.0, conf)

    res = picks_out - line
    dz = np.diff(z)
    dtint = np.diff(picks_out)
    with np.errstate(invalid='ignore', divide='ignore'):
        vint = dz / np.where(dtint > 0, dtint, np.nan)
    fit = DirectArrivalFit(float(b), float(vel), res,
                           float(np.sqrt(np.mean(res[good] ** 2))), vint)
    return FirstBreaks(picks_out, conf_out, fit, picks_out.copy())

def offset_time_correction(fb: FirstBreaks, depths_m: np.ndarray,
                           source_offset_m: float) -> np.ndarray:
    """Remove the straight-ray horizontal-path term to recover vertical time.

    For an offset source the observed arrival is
    ``t(z) = t0 + sqrt(z^2 + X^2)/V``, whose depth slope is
    ``1/(V cos theta)`` - the fitted apparent velocity is *not* V. V and t0
    are estimated by least squares against ``sqrt(z^2 + X^2)``; the
    corrected vertical time is ``t0 + z/V``.
    """
    if source_offset_m <= 0:
        return fb.times_s.copy()
    z = np.asarray(depths_m, float)
    r = np.sqrt(z * z + source_offset_m ** 2)
    rc = r - r.mean()
    tc = fb.times_s - fb.times_s.mean()
    v = float(np.sum(rc * tc) / np.sum(rc * rc))
    vel = 1.0 / v if v > 0 else fb.fit.apparent_velocity_mps
    t0 = float(fb.times_s.mean() - r.mean() / vel)
    return t0 + z / vel

def shift_rows(data: np.ndarray, shifts_samples: np.ndarray) -> np.ndarray:
    """Shift each row by ``shifts_samples`` (positive -> later).

    Fourier fractional shift; samples wrapping through the record are zeroed
    afterwards so first-break flattening cannot introduce wrap-around events.
    """
    d = np.asarray(data, float)
    n, ns = d.shape
    s = np.broadcast_to(np.asarray(shifts_samples, float), (n,)).copy()
    spec = np.fft.rfft(d, axis=1)
    omega = 2 * np.pi * np.fft.rfftfreq(ns, d=1.0)
    phase = np.exp(-1j * omega[None, :] * s[:, None])
    shifted = np.fft.irfft(spec * phase, ns, axis=1)
    # Zero regions that came from beyond either record edge (integer part).
    for i, si in enumerate(s):
        k = int(round(si))
        if k > 0:
            shifted[i, :k] = 0.0
        elif k < 0:
            shifted[i, ns + k:] = 0.0
    return shifted

def align_first_breaks(data: np.ndarray, times_s: np.ndarray, dt_s: float,
                       target_s: float | None = None):
    """Flatten direct arrivals to ``target_s`` (default: shallowest pick)."""
    t = np.asarray(times_s, float)
    if target_s is None:
        target_s = float(np.nanmin(t))
    shifts = (target_s - t) / dt_s
    return shift_rows(data, shifts), float(target_s)
