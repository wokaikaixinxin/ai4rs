# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .icdar2015 import ICDAR15Dataset  # noqa: F401, F403
from .sardet_100k import SAR_Det_Finegrained_Dataset  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403

__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset', 'ICDAR15Dataset', 'SAR_Det_Finegrained_Dataset'
]