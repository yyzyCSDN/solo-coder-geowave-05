from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate
from geowave.well.checkshot import Checkshot

@dataclass(frozen=True)
class SeismicTie:
    shift_s: float
    correlation: float
    scale: float
    target_times_s: np.ndarray
    corridor_on_target_s: np.ndarray

def _resample(trace: np.ndarray, times_s: np.ndarray,
              target_times_s: np.ndarray) -> np.ndarray:
    return np.interp(target_times_s, times_s, trace, left=0.0, right=0.0)

def tie_corridor(times_s: np.ndarray, corridor: np.ndarray,
                 target_times_s: np.ndarray, target_trace: np.ndarray,
                 dt_s: float, max_shift_s: float = 0.1) -> SeismicTie:
    """Cross-correlate the corridor stack with surface seismic or a synthetic.

    Both series are put on the target time axis; the bulk shift, normalized
    correlation and best-fit scale are returned. A large shift usually means
    an unresolved datum/uphole-time or depth-scale issue - check the
    geometry issues attached to the pipeline result.
    """
    target_times_s = np.asarray(target_times_s, float)
    target_trace = np.asarray(target_trace, float)
    a = _resample(np.asarray(corridor, float), np.asarray(times_s, float),
                  target_times_s)
    b = target_trace
    n = min(a.size, b.size)
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    c = correlate(a, b, mode='full')
    lags = np.arange(-n + 1, n)
    m = np.abs(lags) <= round(max_shift_s / dt_s)
    idx = int(np.argmax(c[m]))
    lag = int(lags[m][idx])
    shifted = np.roll(b, lag)
    den = float(np.dot(shifted, shifted))
    scale = float(np.dot(a, shifted) / den) if den else 1.0
    norm = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return SeismicTie(float(lag * dt_s), float(c[m][idx] / norm), scale,
                      target_times_s, _resample(corridor, times_s, target_times_s))

def vsp_checkshots(first_breaks_s: np.ndarray, depths_m: np.ndarray
                   ) -> list[Checkshot]:
    """Convert corrected VSP first breaks to the platform Checkshot format.

    Checkshots in this codebase carry two-way time; direct arrivals are
    one-way, hence the factor of two.
    """
    return [Checkshot(depth_m=float(z), two_way_time_s=float(2.0 * t))
            for z, t in zip(depths_m, first_breaks_s)]
