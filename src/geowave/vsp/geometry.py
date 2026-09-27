from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.vsp.gather import VSPGather

@dataclass(frozen=True)
class GeometryIssue:
    code: str
    severity: str
    detail: str
    impact: str

@dataclass(frozen=True)
class DepthGeometryQC:
    level_count: int
    depth_min_m: float
    depth_max_m: float
    monotonic_with_order: bool
    duplicate_levels: tuple[int, ...]
    spacing_median_m: float
    spacing_cv: float
    dt_s: float
    time_datums_s: tuple[float, ...]
    issues: tuple[GeometryIssue, ...]

def validate_depth_geometry(gather: VSPGather, spacing_cv_warn: float=0.3, datum_tolerance_s: float=1e-6) -> DepthGeometryQC:
    depths = gather.depths_m
    issues: list[GeometryIssue] = []
    mono = bool(np.all(np.diff(depths) > 0))
    if not mono:
        issues.append(GeometryIssue('depth-order', 'warning', '井深未随道序单调递增', '井深顺序存疑：若按道序解释将错层；处理已按深度重排，首波趋势、波场分离与走廊叠加均基于重排后的顺序，若深度登记本身错误，全部深度标定结果随之偏移'))
    order = np.argsort(depths, kind='stable')
    sorted_depths = depths[order]
    dups = tuple((int(order[i + 1]) for i in np.where(np.diff(sorted_depths) <= 0)[0]))
    if dups:
        issues.append(GeometryIssue('duplicate-depth', 'critical', f'重复井深级别(道序): {dups}', '同一井深出现多道：深度域中值滤波孔径与走廊叠加权重失真；处理时每个深度仅保留首见道，其余剔除'))
    uniq = np.unique(sorted_depths)
    spacings = np.diff(uniq)
    med = float(np.median(spacings)) if spacings.size else 0.0
    cv = float(np.std(spacings) / med) if med > 0 else 0.0
    if spacings.size and cv > spacing_cv_warn:
        issues.append(GeometryIssue('irregular-spacing', 'info', f'级间距离散度 cv={cv:.2f}', '级间距严重不均：深度域中值滤波的实际孔径随深度变化，稀疏段波场分离的上下行波泄漏增多'))
    datums = gather.start_times_s
    if len(datums) > 1 and datums[-1] - datums[0] > datum_tolerance_s:
        spread_ms = (datums[-1] - datums[0]) * 1000
        issues.append(GeometryIssue('mixed-time-datum', 'warning', f'各道起始时间不一致，最大差异 {spread_ms:.1f} ms', '采样基准存疑：各道时间零点不统一，首波拾取与叠加前必须归一基准，否则走廊叠加同相轴错位、标定时深关系失真'))
    return DepthGeometryQC(len(gather.levels), float(depths.min()), float(depths.max()), mono, dups, med, cv, gather.dt_s, datums, tuple(issues))
