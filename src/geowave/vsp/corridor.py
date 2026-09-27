from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .firstbreak import shift_rows

@dataclass(frozen=True)
class CorridorStack:
    times_s: np.ndarray
    narrow: np.ndarray
    wide: np.ndarray
    fold_narrow: np.ndarray
    fold_wide: np.ndarray
    narrow_width_s: float
    wide_width_s: float
    first_break_twt_s: np.ndarray

def map_to_twt(upgoing: np.ndarray, first_breaks_s: np.ndarray, dt_s: float,
               start_s: float = 0.0):
    """Map the upgoing panel from recorded time onto two-way time.

    A primary reflection from an interface with two-way time ``tau`` is
    recorded above the interface at ``tau - tb``; delaying the trace by
    ``tb`` places it at ``tau`` on the surface-seismic time base. The
    (removed) direct arrival would land on the first-break TWT line
    ``2*tb``. The common output axis begins at the latest mapped first-break
    line and ends at the earliest shifted record end.
    """
    tb = np.asarray(first_breaks_s, float)
    n = upgoing.shape[1]
    j_start = int(np.ceil(2.0 * float(np.max(tb)) / dt_s))
    j_end = n - 1 + int(np.floor(float(np.min(tb)) / dt_s))
    if j_end <= j_start:
        raise ValueError('record too short to build a common TWT corridor axis')
    shifted = shift_rows(upgoing, tb / dt_s)
    panel = shifted[:, j_start:j_end + 1]
    times = start_s + (j_start + np.arange(panel.shape[1])) * dt_s
    return panel, times, 2.0 * tb

def corridor_stack(upgoing: np.ndarray, first_breaks_s: np.ndarray, dt_s: float,
                   start_s: float = 0.0,
                   narrow_width_s: float = 0.15,
                   wide_width_s: float = 0.35) -> CorridorStack:
    """Narrow- and wide-corridor stacks of the TWT-mapped upgoing panel.

    After the +tb mapping the direct arrival of level i lies on the
    first-break TWT line ``2*tb_i`` and each primary reflection sits at its
    interface TWT ``tau > 2*tb_i`` (the reflected path is always longer than
    the direct one). The corridor keeps samples from the line DOWN to
    ``2*tb_i + width``; later samples - interbed multiples that arrive well
    after the local first break - are muted. The narrow corridor hugs the
    line for the cleanest primaries, the wide one adds fold. Stack fold is
    reported per TWT sample so coverage gaps stay visible.
    """
    if narrow_width_s <= 0 or wide_width_s <= narrow_width_s:
        raise ValueError('require 0 < narrow width < wide width')
    panel, times, fb_line = map_to_twt(upgoing, first_breaks_s, dt_s, start_s)

    def stack_with(width: float) -> tuple[np.ndarray, np.ndarray]:
        keep = ((times[None, :] >= fb_line[:, None] - 1e-9) &
                (times[None, :] <= fb_line[:, None] + width))
        fold = keep.sum(axis=0).astype(float)
        summed = np.sum(np.where(keep, panel, 0.0), axis=0)
        with np.errstate(invalid='ignore'):
            mean = np.where(fold > 0, summed / np.where(fold > 0, fold, 1.0), 0.0)
        return mean, fold

    narrow, f_n = stack_with(narrow_width_s)
    wide, f_w = stack_with(wide_width_s)
    return CorridorStack(times, narrow, wide, f_n, f_w,
                         narrow_width_s, wide_width_s, fb_line)
