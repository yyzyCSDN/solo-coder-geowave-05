from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from geowave.model.traces import Trace
from geowave.well.checkshot import Checkshot
from geowave.vsp.gather import VSPGather
from geowave.vsp.geometry import DepthGeometryQC, validate_depth_geometry
from geowave.vsp.first_break import FirstBreakAlignment, pick_first_breaks, align_on_first_break
from geowave.vsp.separation import WavefieldSeparation, median_separation
from geowave.vsp.corridor import CorridorStackResult, corridor_stack
from geowave.vsp.tie import SurfaceTie, CheckshotTie, tie_to_surface, checkshot_calibration

@dataclass(frozen=True)
class VSPProcessingResult:
    geometry: DepthGeometryQC
    alignment: FirstBreakAlignment
    separation: WavefieldSeparation
    corridor: CorridorStackResult
    surface_tie: SurfaceTie | None
    checkshot_tie: CheckshotTie | None
    impacts: tuple[str, ...]
    grade: str

def process_vsp(gather: VSPGather, surface: Trace | None=None, checkshots: list[Checkshot] | None=None, *, median_window: int=5, corridor_s: tuple[float, float]=(0.0, 0.6), max_tie_shift_s: float=0.2, polarity_margin: float=0.05, tie_shift_tol_s: float=0.01, checkshot_tol_s: float=0.004) -> VSPProcessingResult:
    impacts: list[str] = []
    qc = validate_depth_geometry(gather)
    impacts.extend(i.impact for i in qc.issues)
    work = gather
    if not qc.monotonic_with_order:
        work = work.sorted_by_depth()
    if qc.duplicate_levels:
        seen: set[float] = set()
        kept = []
        for lv in work.levels:
            if lv.depth_m in seen:
                continue
            seen.add(lv.depth_m)
            kept.append(lv)
        work = VSPGather(tuple(kept), work.name + '-dedup', work.datum)
    if len(work.start_times_s) > 1:
        work, shifts = work.unify_time_datum()
        impacts.append(f'采样基准已统一：各道按最早起始时间对齐，最大平移 {float(np.max(shifts)) * 1000:.1f} ms；若该差异源于采集基准设置错误而非记录延迟，首波时间与走廊叠加 TWT 存在同量级系统偏差')
    alignment = pick_first_breaks(work)
    if alignment.low_confidence_levels:
        impacts.append(f'{len(alignment.low_confidence_levels)} 个级别首波偏离深度趋势或置信度低(级别 {alignment.low_confidence_levels})：这些深度附近下行波去除不净，走廊叠加可能混入下行波能量')
    aligned = align_on_first_break(work, alignment)
    separation = median_separation(aligned, median_window)
    corridor = corridor_stack(separation.upgoing, alignment, corridor_s)
    surface_tie = None
    if surface is not None:
        surface_tie = tie_to_surface(corridor, surface, max_tie_shift_s)
        if surface_tie.polarity_flipped:
            if surface_tie.polarity_margin >= polarity_margin:
                corridor = replace(corridor, stack=-corridor.stack)
                impacts.append('极性与地面地震相反：走廊叠加已反转极性以匹配地面地震；若地面地震的极性约定变更，该反转需同步回滚')
            else:
                impacts.append('极性存疑：走廊叠加与地面地震正/反相关系数接近，未做反转；与地面地震对比解释时两者极性可能相反')
        if abs(surface_tie.shift_s) > tie_shift_tol_s:
            impacts.append(f'走廊叠加与地面地震存在 {surface_tie.shift_s * 1000:.1f} ms 时移：多为采样基准或首波拾取系统偏差所致，对比解释前应按该时移对齐')
    checkshot_tie = None
    if checkshots:
        checkshot_tie = checkshot_calibration(work.depths_m, alignment.picks_s, checkshots)
        if abs(checkshot_tie.mean_s) > checkshot_tol_s:
            direction = '偏晚' if checkshot_tie.mean_s > 0 else '偏早'
            impacts.append(f'首波与井震标定存在系统偏差 {checkshot_tie.mean_s * 1000:.1f} ms(VSP {direction})：疑似采样基准/静校正问题，走廊叠加 TWT 同向偏移，与地面地震及时深关系对比时应补偿')
    severities = [i.severity for i in qc.issues]
    n_low = len(alignment.low_confidence_levels)
    grade = 'A'
    if 'warning' in severities or n_low:
        grade = 'B'
    if n_low > max(2, len(work.levels) // 5):
        grade = 'C'
    if 'critical' in severities:
        grade = 'D'
    return VSPProcessingResult(qc, alignment, separation, corridor, surface_tie, checkshot_tie, tuple(impacts), grade)
