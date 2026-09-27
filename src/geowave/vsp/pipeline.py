from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from .model import VSPGather, Issue
from .geometry import check_geometry, regularize_levels, GeometryQC
from .firstbreak import (FirstBreaks, pick_first_breaks,
                         offset_time_correction, align_first_breaks,
                         shift_rows)
from .polarity import PolarityQC, check_polarity, apply_polarity_fix
from .separation import Wavefields, median_separate, fk_separate
from .corridor import CorridorStack, corridor_stack
from .tie import SeismicTie, tie_corridor, vsp_checkshots

@dataclass
class VSPResult:
    gather: VSPGather
    geometry: GeometryQC
    depths_m: np.ndarray                  # regular grid actually used
    panel: np.ndarray                     # sorted, regularized, polarity-fixed
    first_breaks: FirstBreaks
    corrected_first_breaks_s: np.ndarray
    flattened: np.ndarray
    flatten_target_s: float
    polarity: PolarityQC
    polarity_flipped: list[int]
    wavefields: Wavefields
    corridor: CorridorStack
    issues: list[Issue] = field(default_factory=list)
    surface_tie: SeismicTie | None = None
    synthetic_tie: SeismicTie | None = None

    def checkshots(self) -> list:
        return vsp_checkshots(self.corrected_first_breaks_s, self.depths_m)

    def issues_by_severity(self) -> dict[str, list[Issue]]:
        out: dict[str, list[Issue]] = {'error': [], 'warning': [], 'info': []}
        for it in self.issues:
            out[it.severity].append(it)
        return out

    def impact_report(self) -> str:
        """Human-readable summary; every doubt and its downstream impact."""
        lines = [f'VSP processing report: {self.gather.name}',
                 f'levels={self.depths_m.size}  '
                 f'depth {self.depths_m[0]:.1f}-{self.depths_m[-1]:.1f} m  '
                 f'datum={self.gather.datum}  '
                 f'separation={self.wavefields.method}']
        counts = Counter(it.severity for it in self.issues)
        lines.append('issues: ' +
                     ', '.join(f'{k}={counts.get(k, 0)}'
                               for k in ('error', 'warning', 'info')))
        if self.polarity_flipped:
            lines.append(f'polarity-normalized levels: {self.polarity_flipped}')
        v = self.first_breaks.fit.apparent_velocity_mps
        lines.append(f'apparent first-break velocity: {v:.0f} m/s, '
                     f'rms pick residual '
                     f'{self.first_breaks.fit.rms_residual_s * 1e3:.1f} ms')
        if self.surface_tie is not None:
            t = self.surface_tie
            lines.append(f'surface seismic tie: shift={t.shift_s * 1e3:.1f} ms '
                         f'correlation={t.correlation:.3f}')
        if self.synthetic_tie is not None:
            t = self.synthetic_tie
            lines.append(f'synthetic tie: shift={t.shift_s * 1e3:.1f} ms '
                         f'correlation={t.correlation:.3f}')
        for sev in ('error', 'warning', 'info'):
            for it in self.issues_by_severity()[sev]:
                lines.append(f'  [{sev.upper()}] ({it.stage}) {it.code}: '
                             f'{it.message}')
                lines.append(f'      impact: {it.impact}')
        return '\n'.join(lines)

def _first_break_issues(fb: FirstBreaks) -> list[Issue]:
    out: list[Issue] = []
    weak = int(np.sum(fb.confidence < 0.15))
    if weak:
        out.append(Issue(
            'firstbreak-lowconf', 'warning',
            f'{weak} level(s) had no confident AIC first break and were '
            'assigned the fitted t-z line.',
            'Flattening at those levels follows the average velocity: '
            'downgoing energy leaks into the upgoing panel and the corridor '
            'stack near those depths; inspect picks before any amplitude use.',
            'first-break'))
    if fb.fit.rms_residual_s > 0.01:
        out.append(Issue(
            'firstbreak-scatter', 'warning',
            f'first-break picks scatter {fb.fit.rms_residual_s * 1e3:.0f} ms '
            'rms around the fitted line.',
            'Residual statics on the direct arrival reduce median-filter '
            'separation quality and widen the corridor event; consider '
            'manual repicking or a velocity gradient.',
            'first-break'))
    if fb.fit.intercept_s < -1e-3:
        out.append(Issue(
            'uphole-time-negative', 'warning',
            f'fitted source (uphole) time is {fb.fit.intercept_s * 1e3:.1f} ms, '
            'negative beyond tolerance.',
            'A negative t(0) usually means the time break or datum is wrong. '
            'TWT of every corridor event and the surface-seismic tie inherit '
            'the same bulk error until corrected.',
            'first-break'))
    dv = fb.fit.interval_velocities_mps
    bad = np.sum((dv < 1000.0) | (dv > 8000.0) | ~np.isfinite(dv))
    if bad:
        out.append(Issue(
            'interval-velocity-unphysical', 'warning',
            f'{int(bad)} interval(s) give interval velocities outside '
            '1000-8000 m/s.',
            'Bad intervals point to swapped picks or depths; interval '
            'velocity products and any drift/tie QC at those intervals are '
            'not trustworthy.',
            'first-break'))
    return out

def run_vsp_pipeline(vsp: VSPGather, *,
                     method: str = 'median',
                     median_half_width: int = 3,
                     search_s: tuple[float, float] | None = None,
                     fix_polarity: bool = True,
                     target_times_s: np.ndarray | None = None,
                     surface_seismic: np.ndarray | None = None,
                     synthetic: np.ndarray | None = None,
                     max_tie_shift_s: float = 0.1) -> VSPResult:
    """End-to-end zero/offset VSP workflow.

    geometry check -> even-depth regularization -> first-break pick/fit and
    offset correction -> polarity check (fixed by default, never silently)
    -> flatten + up/down separation -> TWT mapping + narrow/wide corridor
    stacks -> ties with surface seismic and synthetic. All doubts are
    returned on ``result.issues`` and via :meth:`VSPResult.impact_report`.
    """
    issues: list[Issue] = []

    qc = check_geometry(vsp)
    issues.extend(qc.issues)
    depths, sorted_panel = regularize_levels(qc)

    if method not in ('median', 'fk'):
        raise ValueError("method must be 'median' or 'fk'")

    fb = pick_first_breaks(depths, sorted_panel, vsp.dt_s,
                           start_s=vsp.start_s, search_s=search_s)
    issues.extend(_first_break_issues(fb))
    corrected = offset_time_correction(fb, depths, vsp.source_offset_m)
    if vsp.source_offset_m > 0 and fb.fit.apparent_velocity_mps > 0:
        shift0 = (np.hypot(depths[0], vsp.source_offset_m) - depths[0]) \
                 / fb.fit.apparent_velocity_mps
        if shift0 > vsp.dt_s:
            issues.append(Issue(
                'offset-correction', 'info',
                f'straight-ray offset correction up to {shift0 * 1e3:.0f} ms '
                '(shallowest level) applied to first breaks.',
                'Correction assumes constant velocity and straight rays; '
                'near-surface refraction and anisotropy would leave a residual '
                'moveout that flattens poorly and biases the shallow corridor.',
                'first-break'))

    # Flatten on corrected (vertical) first-break times for QC and median path.
    flattened, target = align_first_breaks(sorted_panel, corrected, vsp.dt_s)

    pol, pol_issues = check_polarity(flattened, target, vsp.dt_s)
    issues.extend(pol_issues)
    polarity_flipped: list[int] = []
    work = sorted_panel
    flat_work = flattened
    if fix_polarity and any(pol.signs < 0):
        work, polarity_flipped = apply_polarity_fix(sorted_panel, pol)
        flat_work, _ = align_first_breaks(work, corrected, vsp.dt_s)
        issues.append(Issue(
            'polarity-applied', 'info',
            f'flipped {len(polarity_flipped)} reversed level(s): '
            f'{polarity_flipped}',
            'Levels now share the pilot polarity. If the pilot itself was '
            'wrong (majority reversed relative to the surface convention), '
            'the whole corridor stack carries the opposite sign - confirm '
            'the polarity code against the acquisition report.',
            'polarity'))
    elif any(pol.signs < 0):
        issues.append(Issue(
            'polarity-unfixed', 'error',
            'reversed levels detected but fix_polarity=False.',
            'Wavefield separation and corridor stack below retain mixed '
            'polarity; amplitudes and the tie sign are not interpretable.',
            'polarity'))

    if not vsp.polarity_verified:
        issues.append(Issue(
            'polarity-convention-unverified', 'warning',
            f'recording polarity code ({vsp.polarity_code:+d}) not confirmed '
            'against acquisition report/surface-seismic convention.',
            'Relative polarity inside the VSP is corrected, but the absolute '
            'sign relative to surface seismic is unverified; the tie '
            'correlation amplitude is reliable but a global sign may be off.',
            'polarity'))

    dz = float(depths[1] - depths[0])
    if method == 'median':
        fields = median_separate(flat_work, median_half_width)
        # Separation ran on first-break-flattened data; undo the flattening
        # so the upgoing panel is back on the recorded time axis, which is
        # what the TWT corridor mapping expects.
        up_raw = shift_rows(fields.upgoing,
                            (corrected - target) / vsp.dt_s)
        down_raw = shift_rows(fields.downgoing,
                              (corrected - target) / vsp.dt_s)
        fields = Wavefields(down_raw, up_raw, fields.method)
    else:
        fields = fk_separate(work, vsp.dt_s, dz,
                             fb.fit.apparent_velocity_mps)
    up_times = corrected

    corr = corridor_stack(fields.upgoing, up_times, vsp.dt_s,
                          start_s=vsp.start_s)

    # Coverage sanity: per TWT sample only levels whose first-break TWT line
    # falls within one corridor width contribute, so expected narrow fold is
    # width / (2*dz/v); check against that, not the total level count.
    v_fit = fb.fit.apparent_velocity_mps
    expected_fold = corr.narrow_width_s / max(2.0 * dz / v_fit, 1e-9)
    if np.max(corr.fold_narrow) < max(3, 0.5 * expected_fold):
        issues.append(Issue(
            'corridor-low-fold', 'warning',
            f'narrow-corridor peak fold is {int(np.max(corr.fold_narrow))} '
            f'against ~{expected_fold:.0f} expected from geometry.',
            'Low fold means the record is short relative to receiver span or '
            'spacing is uneven; deep corridor events are poorly constrained '
            'and the tie there should not be over-interpreted.',
            'corridor'))

    result = VSPResult(vsp, qc, depths, work, fb, corrected, flat_work,
                       target, pol, polarity_flipped, fields, corr, issues)

    if target_times_s is not None and surface_seismic is not None:
        result.surface_tie = tie_corridor(
            corr.times_s, corr.narrow, np.asarray(target_times_s, float),
            np.asarray(surface_seismic, float), vsp.dt_s, max_tie_shift_s)
        if abs(result.surface_tie.shift_s) > 0.03:
            issues.append(Issue(
                'surface-tie-shift', 'warning',
                f'corridor stack needs a {result.surface_tie.shift_s * 1e3:.0f} '
                'ms bulk shift to match surface seismic.',
                'A bulk shift points at datum/uphole-time mismatch, a timing '
                'base error or unresolved depth scale; see geometry issues. '
                'Do not apply the shift blindly without identifying its cause.',
                'tie'))
        if result.surface_tie.correlation < 0.3:
            issues.append(Issue(
                'surface-tie-poor', 'warning',
                f'surface-seismic tie correlation only '
                f'{result.surface_tie.correlation:.2f}.',
                'Weak tie may follow residual multiples inside the corridor, '
                'wavelet phase differences or the unverified polarity code; '
                'horizon-level calibration is not supported at this quality.',
                'tie'))

    if target_times_s is not None and synthetic is not None:
        result.synthetic_tie = tie_corridor(
            corr.times_s, corr.narrow, np.asarray(target_times_s, float),
            np.asarray(synthetic, float), vsp.dt_s, max_tie_shift_s)
        if abs(result.synthetic_tie.shift_s) > 0.03:
            issues.append(Issue(
                'synthetic-tie-shift', 'warning',
                f'synthetic tie requires {result.synthetic_tie.shift_s * 1e3:.0f}'
                ' ms bulk shift; check the checkshot/depth-time curve and '
                'wavelet timing before stretching to fit.',
                'The shift is carried into any horizon calibration; a sonic '
                'drift of this size must be reconciled with the VSP checkshots.',
                'tie'))

    return result
