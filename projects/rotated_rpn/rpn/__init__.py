from .avvg import AVVGDetDataset
from .dior_r_rsvg import DIORRRSVGDetDataset
from .vrsbench import VRSBenchDetDataset
from .obb_recall_metric import OBBRecallMetric


__all__ = [
    'AVVGDetDataset',
    'DIORRRSVGDetDataset',
    'VRSBenchDetDataset',
    'OBBRecallMetric'
]