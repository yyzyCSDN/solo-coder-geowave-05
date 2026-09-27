from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.signal.statics import apply_static
from geowave.vsp.gather import VSPGather
from geowave.vsp.first_break import FirstBreakAlignment

@dataclass(frozen=True)
class CorridorStackResult:
    twt_s: np.ndarray
    stack: np.ndarray
    fold: np.ndarray
    corridor_s: tuple[float, float]
    levels_used: int

def corridor_stack(upgoing_aligned: VSPGather, alignment: FirstBreakAlignment, corridor_s: tuple[float, float]=(0.0, 0.6)) -> CorridorStackResult:
    dt = upgoing_aligned.dt_s
    start = upgoing_aligned.levels[0].trace.header.start_s
    n = upgoing_aligned.n_samples
    times = start + np.arange(n) * dt
    acc = np.zeros(n)
    fold = np.zeros(n)
    used = 0
    for lv, pick in zip(upgoing_aligned.levels, alignment.picks_s):
        twt_trace = apply_static(lv.trace, 2 * float(pick) - alignment.reference_s)
        lo = 2 * float(pick) + corridor_s[0]
        hi = 2 * float(pick) + corridor_s[1]
        mask = (times >= lo - 1e-9) & (times <= hi + 1e-9)
        if not mask.any():
            continue
        acc[mask] += twt_trace.samples[mask]
        fold[mask] += 1
        used += 1
    stack = np.divide(acc, fold, out=np.zeros_like(acc), where=fold > 0)
    return CorridorStackResult(times, stack, fold, (float(corridor_s[0]), float(corridor_s[1])), used)
