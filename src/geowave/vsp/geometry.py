from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .model import VSPGather, Issue

@dataclass
class GeometryQC:
    depths_m: np.ndarray
    order: np.ndarray               # acquisition-row -> depth-sorted-row mapping
    receiver_ids: list[str]
    raw_data: np.ndarray
    issues: list[Issue] = field(default_factory=list)

    @property
    def n_levels(self) -> int:
        return int(self.depths_m.size)

    def median_spacing_m(self) -> float:
        if self.n_levels < 2:
            return 0.0
        return float(np.median(np.diff(self.depths_m)))

def check_geometry(vsp: VSPGather, spacing_tol: float = 0.02) -> GeometryQC:
    """Validate depth order/duplicates, level spacing and the sampling datum.

    Depth rows are always returned sorted by depth; when the acquisition
    order disagrees with depth order an issue is emitted and every later
    product carries that doubt. Regularization to an even depth grid happens
    separately (see :func:`regularize_levels`); here only irregular spacing
    is reported.
    """
    issues: list[Issue] = []
    z = vsp.depths_m
    n = z.size

    if n < 2:
        raise ValueError('VSP needs at least two receiver levels')
    if np.any(~np.isfinite(z)):
        raise ValueError('non-finite receiver depth')

    uniq, counts = np.unique(z, return_counts=True)
    dup = uniq[counts > 1]
    if dup.size:
        issues.append(Issue(
            'duplicate-depth', 'error',
            f'{dup.size} receiver depth(s) are recorded more than once: '
            f'{np.round(dup, 3).tolist()}',
            'Stacks at the same depth are ambiguous; duplicate rows were '
            'removed by keeping the first acquisition row, check the survey '
            'sheet before trusting amplitudes.',
            'geometry'))

    order = np.argsort(z, kind='stable')
    increasing = np.all(np.diff(z) > 0)
    if not increasing:
        issues.append(Issue(
            'depth-order', 'error',
            'receiver levels are not stored in increasing measured depth '
            '(duplicate or decreasing depth found).',
            'Depth order has been corrected for all downstream products. '
            'If the depth column is actually mislabeled rather than merely '
            'shuffled, first-break times, interval velocities and corridor '
            'geometry will be wrong by the same amount.',
            'geometry'))
    elif not np.array_equal(order, np.arange(n)):
        issues.append(Issue(
            'depth-order', 'warning',
            'trace order disagrees with increasing depth; rows were reordered.',
            'Processing now follows depth order. Verify the depth column came '
            'from the same survey as the trace headers, otherwise up/down '
            'dips in wavefield separation are reversed.',
            'geometry'))

    zs = z[order]
    dz = np.diff(zs)
    med = float(np.median(dz))
    if med > 0:
        bad = np.abs(dz - med) / med > spacing_tol
        if np.any(bad):
            k = int(np.argmax(bad))
            issues.append(Issue(
                'irregular-spacing', 'warning',
                f'depth spacing varies (median {med:.2f} m; worst gap '
                f'{dz[k]:.2f} m between {zs[k]:.1f} and {zs[k + 1]:.1f} m).',
                'f-k separation assumes an even depth grid. Use the median '
                'filter path or the regularized panel; otherwise fan masks '
                'leak upgoing energy into the downgoing wave.',
                'geometry'))

    if abs(vsp.source_offset_m) > max(0.0, med):
        issues.append(Issue(
            'offset-vsp', 'info',
            f'source is {vsp.source_offset_m:.1f} m from the wellhead; this is '
            'an offset VSP, not zero-offset.',
            'Direct-arrival times contain a horizontal slowness term. TWT and '
            'interval velocities below are straight-ray corrected; steep dips '
            'and anisotropy are not handled and corridor ties may sit late.',
            'geometry'))

    if not vsp.depth_verified:
        issues.append(Issue(
            'depth-unverified', 'warning',
            'receiver depths have not been verified against the deviation/survey sheet.',
            'An unconfirmed measured-depth scale shifts first-break picks and '
            'the corridor-stack time axis proportionally; the tie with surface '
            'seismic and checkshots cannot be called until depth is confirmed.',
            'geometry'))

    if not vsp.datum_verified:
        issues.append(Issue(
            'datum-unverified', 'warning',
            f'sampling datum ({vsp.datum}) is not confirmed against surface seismic '
            'datum/checkshot reference.',
            'A datum or uphole-time mismatch (e.g. KB vs ground level) produces '
            'a static on every first break and a constant bulk shift of the '
            'corridor stack relative to surface seismic.',
            'geometry'))

    # Drop duplicate depths, keeping the first acquisition row.
    keep = np.ones(n, dtype=bool)
    seen: set[float] = set()
    for i, zi in enumerate(z):
        if zi in seen:
            keep[i] = False
        seen.add(zi)
    order = order[np.isin(z[order], z[keep])]

    return GeometryQC(z[order], order,
                      [vsp.receiver_ids[i] for i in order],
                      vsp.data[order], issues)

def regularize_levels(qc: GeometryQC) -> tuple[np.ndarray, np.ndarray]:
    """Resample sorted levels onto an even depth grid (linear interpolation)."""
    z = qc.depths_m
    reg = np.linspace(z[0], z[-1], z.size)
    d = np.empty_like(qc.raw_data)
    for j in range(qc.raw_data.shape[1]):
        d[:, j] = np.interp(reg, z, qc.raw_data[:, j])
    return reg, d
