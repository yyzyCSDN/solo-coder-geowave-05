from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from geowave.model.traces import Trace, TraceSet
from geowave.signal.statics import apply_static

@dataclass(frozen=True)
class VSPLevel:
    depth_m: float
    trace: Trace

@dataclass
class VSPGather:
    levels: tuple[VSPLevel, ...]
    name: str = 'vsp'
    datum: str = 'unknown'

    def __post_init__(self):
        self.levels = tuple(self.levels)
        if not self.levels:
            raise ValueError('empty vsp gather')
        dt = self.levels[0].trace.header.dt_s
        n = self.levels[0].trace.samples.size
        for lv in self.levels[1:]:
            if abs(lv.trace.header.dt_s - dt) > 1e-12:
                raise ValueError('mixed sample intervals')
            if lv.trace.samples.size != n:
                raise ValueError('mixed trace lengths')

    @property
    def depths_m(self) -> np.ndarray:
        return np.asarray([lv.depth_m for lv in self.levels], float)

    @property
    def dt_s(self) -> float:
        return self.levels[0].trace.header.dt_s

    @property
    def n_samples(self) -> int:
        return self.levels[0].trace.samples.size

    @property
    def start_times_s(self) -> tuple[float, ...]:
        return tuple(sorted({lv.trace.header.start_s for lv in self.levels}))

    def to_matrix(self) -> np.ndarray:
        return np.vstack([lv.trace.samples for lv in self.levels])

    def with_traces(self, traces, name: str | None=None) -> 'VSPGather':
        traces = list(traces)
        if len(traces) != len(self.levels):
            raise ValueError('trace count mismatch')
        return VSPGather(tuple(VSPLevel(lv.depth_m, tr) for lv, tr in zip(self.levels, traces)), name or self.name, self.datum)

    def sorted_by_depth(self) -> 'VSPGather':
        return VSPGather(tuple(sorted(self.levels, key=lambda lv: lv.depth_m)), self.name + '-zsorted', self.datum)

    def unify_time_datum(self) -> tuple['VSPGather', np.ndarray]:
        starts = np.asarray([lv.trace.header.start_s for lv in self.levels], float)
        base = float(starts.min())
        shifts = starts - base
        traces = []
        for lv, sh in zip(self.levels, shifts):
            tr = apply_static(lv.trace, float(sh)) if sh > 1e-12 else lv.trace
            traces.append(Trace(replace(tr.header, start_s=base), tr.samples, dict(tr.tags)))
        return (self.with_traces(traces, self.name + '-datum'), shifts)

def from_trace_set(dataset: TraceSet, depths_m, datum: str='unknown') -> VSPGather:
    if isinstance(depths_m, dict):
        levels = [VSPLevel(float(depths_m[t.header.trace_id]), t) for t in dataset.traces]
    else:
        depths = list(depths_m)
        if len(depths) != len(dataset.traces):
            raise ValueError('depth count mismatch')
        levels = [VSPLevel(float(z), t) for z, t in zip(depths, dataset.traces)]
    return VSPGather(tuple(levels), dataset.name, datum)
