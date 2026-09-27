from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .model import Issue

@dataclass(frozen=True)
class PolarityQC:
    signs: np.ndarray               # +1 / -1 relative to pilot
    correlation: np.ndarray         # pilot correlation of the first-break lobe
    flipped_levels: list[int]       # indices normalized by apply_polarity_fix

def pilot_from_levels(flattened: np.ndarray, pick_time_s: float, dt_s: float,
                      window_s: float = 0.06) -> np.ndarray:
    """Stack a short window around the flattened first break as the pilot."""
    i0 = max(0, int((pick_time_s - window_s) / dt_s))
    i1 = min(flattened.shape[1], int((pick_time_s + window_s) / dt_s))
    p = np.mean(flattened[:, i0:i1], axis=0)
    return p / (np.max(np.abs(p)) + 1e-12)

def check_polarity(flattened: np.ndarray, pick_time_s: float, dt_s: float,
                   window_s: float = 0.06, min_correlation: float = 0.2
                   ) -> tuple[PolarityQC, list[Issue]]:
    """Correlate each level's first-break lobe with the stacked pilot.

    A negative correlation means the level is reversed relative to the
    majority. Polarity is never flipped silently: the caller decides after
    reading the issue impact.
    """
    i0 = max(0, int((pick_time_s - window_s) / dt_s))
    i1 = min(flattened.shape[1], int((pick_time_s + window_s) / dt_s))
    pilot = pilot_from_levels(flattened, pick_time_s, dt_s, window_s)
    signs = np.ones(flattened.shape[0], dtype=int)
    corr = np.zeros(flattened.shape[0])
    issues: list[Issue] = []
    for i in range(flattened.shape[0]):
        x = flattened[i, i0:i1]
        if np.std(x) < 1e-12:
            corr[i] = 0.0
            continue
        c = float(np.corrcoef(x, pilot)[0, 1])
        corr[i] = c
        if c < -min_correlation:
            signs[i] = -1
    bad = [int(i) for i in np.where(signs < 0)[0]]
    if bad:
        issues.append(Issue(
            'polarity-reversed', 'error',
            f'{len(bad)} level(s) have a first-break lobe opposite to the '
            f'stacked pilot: {bad}',
            'Reversed levels must be flipped before median/f-k separation, '
            'otherwise the downgoing estimate is corrupted, deconvolution '
            'destructively interferes and the corridor stack amplitude (and '
            'any amplitude-based tie) is unreliable at those levels.',
            'polarity'))
    weak = int(np.sum(np.abs(corr) < min_correlation))
    if weak:
        issues.append(Issue(
            'polarity-weak', 'warning',
            f'{weak} level(s) correlate weakly (|c| < {min_correlation:g}) '
            'with the first-break pilot.',
            'Polarity at these levels is assumed same-sense; a wrong call '
            'leaks downgoing energy into the upgoing panel and degrades the '
            'shallow corridor stack.',
            'polarity'))
    return PolarityQC(signs, corr, []), issues

def apply_polarity_fix(data: np.ndarray, qc: PolarityQC) -> tuple[np.ndarray, list[int]]:
    """Flip levels whose first-break lobe opposes the pilot."""
    out = data.copy()
    flipped = [int(i) for i in np.where(qc.signs < 0)[0]]
    for i in flipped:
        out[i] = -out[i]
    return out, flipped
