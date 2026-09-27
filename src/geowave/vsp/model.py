from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SEVERITIES = ('error', 'warning', 'info')

@dataclass(frozen=True)
class Issue:
    """A problem or unverified assumption found during VSP processing.

    ``impact`` must state how downstream products (first-break timing,
    wavefield separation, corridor-stack TWT, well tie) are affected,
    so doubts about depth order, sampling datum or polarity travel with
    the result instead of being silently corrected.
    """
    code: str
    severity: str
    message: str
    impact: str
    stage: str = ''

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f'severity must be one of {SEVERITIES}')

@dataclass
class VSPGather:
    """Borehole geophone levels stored as a 2-D panel (levels x samples).

    Rows are accepted in acquisition order; the geometry stage checks and
    fixes depth ordering. Depths are measured depths below ``datum`` and
    increase downward. ``start_s`` is the time of the first sample relative
    to the recorded time break.
    """
    depths_m: np.ndarray
    data: np.ndarray
    dt_s: float
    start_s: float = 0.0
    source_offset_m: float = 0.0
    receiver_ids: list[str] = field(default_factory=list)
    datum: str = 'KB'
    datum_verified: bool = True
    depth_verified: bool = True
    polarity_code: int = 1
    polarity_verified: bool = True
    name: str = 'vsp'

    def __post_init__(self):
        z = np.asarray(self.depths_m, float)
        d = np.asarray(self.data, float)
        if d.ndim != 2:
            raise ValueError('VSP data must be 2-D (receiver levels x samples)')
        if z.shape[0] != d.shape[0]:
            raise ValueError('depth count must match receiver-level count')
        if self.dt_s <= 0:
            raise ValueError('sample interval must be positive')
        if self.polarity_code not in (1, -1):
            raise ValueError('polarity_code must be +1 or -1')
        self.depths_m = z
        self.data = d
        if not self.receiver_ids:
            self.receiver_ids = [f'r{i}' for i in range(d.shape[0])]
        if len(self.receiver_ids) != d.shape[0]:
            raise ValueError('receiver id count must match receiver-level count')

    @property
    def n_levels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def times(self) -> np.ndarray:
        return self.start_s + np.arange(self.n_samples) * self.dt_s

    @property
    def end_s(self) -> float:
        return self.start_s + (self.n_samples - 1) * self.dt_s
