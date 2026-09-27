from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter
from geowave.vsp.gather import VSPGather

@dataclass(frozen=True)
class WavefieldSeparation:
    downgoing: VSPGather
    upgoing: VSPGather
    method: str
    downgoing_energy: float
    upgoing_energy: float

def median_separation(aligned: VSPGather, window_levels: int=5) -> WavefieldSeparation:
    m = aligned.to_matrix()
    w = max(3, int(window_levels) | 1)
    down = median_filter(m, size=(w, 1), mode='nearest')
    up = m - down
    down_g = aligned.with_traces((lv.trace.copy_with(d, wavefield='downgoing') for lv, d in zip(aligned.levels, down)), aligned.name + '-down')
    up_g = aligned.with_traces((lv.trace.copy_with(u, wavefield='upgoing') for lv, u in zip(aligned.levels, up)), aligned.name + '-up')
    return WavefieldSeparation(down_g, up_g, f'median:{w}', float(np.sum(down ** 2)), float(np.sum(up ** 2)))
