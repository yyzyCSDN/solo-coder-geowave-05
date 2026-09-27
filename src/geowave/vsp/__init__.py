"""Zero/offset VSP processing.

Workflow: geometry/depth QC -> first-break pick, fit and alignment ->
polarity QC -> up/down wavefield separation -> TWT mapping and corridor
stack -> tie to surface seismic and well synthetics.

Doubts about depth order, sampling datum or polarity are represented as
:class:`~geowave.vsp.model.Issue` records with an explicit downstream
``impact`` and travel on :class:`~geowave.vsp.pipeline.VSPResult`.
"""
from .model import VSPGather, Issue
from .geometry import check_geometry, regularize_levels, GeometryQC
from .firstbreak import (FirstBreaks, DirectArrivalFit, pick_first_breaks,
                         offset_time_correction, align_first_breaks,
                         shift_rows)
from .polarity import PolarityQC, check_polarity, apply_polarity_fix
from .separation import Wavefields, median_separate, fk_separate
from .corridor import CorridorStack, corridor_stack, map_to_twt
from .tie import SeismicTie, tie_corridor, vsp_checkshots
from .pipeline import VSPResult, run_vsp_pipeline

__all__ = [
    'VSPGather', 'Issue',
    'check_geometry', 'regularize_levels', 'GeometryQC',
    'FirstBreaks', 'DirectArrivalFit', 'pick_first_breaks',
    'offset_time_correction', 'align_first_breaks', 'shift_rows',
    'PolarityQC', 'check_polarity', 'apply_polarity_fix',
    'Wavefields', 'median_separate', 'fk_separate',
    'CorridorStack', 'corridor_stack', 'map_to_twt',
    'SeismicTie', 'tie_corridor', 'vsp_checkshots',
    'VSPResult', 'run_vsp_pipeline',
]
