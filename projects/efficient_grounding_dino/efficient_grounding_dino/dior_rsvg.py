import os
import os.path as osp
from typing import Dict, List
import xml.etree.ElementTree as ET
import mmengine
from mmengine.dataset import BaseDataset
import numpy as np


class DIORRSVGDataset(BaseDataset):
    """Visual Grounding in Remote Sensing dataset.

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): Prefix for training data.
        split_file (str): Split file path.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """
    METAINFO = {
        'classes':
        ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
         'chimney', 'expressway-service-area', 'expressway-toll-station',
         'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
         'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
         'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42),
                    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0),
                    (120, 166, 157)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 split_file: str,
                 data_prefix: Dict,
                 **kwargs):
        self.split_file = split_file

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs)

    def _join_prefix(self):
        if isinstance(self.split_file, str):
            self.split_file = [self.split_file]

        assert len(self.split_file) > 0, 'there is no file in split file'

        split_file_list = []
        for file in self.split_file:
            if not mmengine.is_abs(file) and file:
                file = osp.join(self.data_root, file)
            split_file_list.append(file)

        self.split_file = split_file_list

        return super()._join_prefix()

    def load_data_list(self) -> List[dict]:
        """Load DIOR-RSVG annotation list."""
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }
        data_list = []

        anno_files = sorted(
            osp.join(self.ann_file, file_name)
            for file_name in os.listdir(self.ann_file)
            if file_name.endswith('.xml')
        )

        img_prefix = self.data_prefix['img_path']

        indices = set()
        for split_file in self.split_file:
            with open(split_file, 'r') as f:
                indices.update(int(line.strip()) for line in f if line.strip())

        count = 0

        for anno_path in anno_files:
            root = ET.parse(anno_path).getroot()

            filename = root.findtext('filename')
            image_path = osp.join(img_prefix, filename)
            img_id = osp.splitext(filename)[0]

            size = root.find('size')
            width = int(size.findtext('width'))
            height = int(size.findtext('height'))

            for member in root.findall('object'):

                if count in indices:
                    bbox = member.find('bndbox')
                    if bbox is None:
                        count += 1
                        continue

                    box = np.array([
                        float(bbox.findtext('xmin')),
                        float(bbox.findtext('ymin')),
                        float(bbox.findtext('xmax')),
                        float(bbox.findtext('ymax')),
                    ], dtype=np.float32)
                    bbox_label = cls_map[member.find('name').text.lower()]

                    text = member.findtext('description', default='')
                    if len(text) <= 1:
                        count += 1
                        continue

                    data_list.append({
                        'img_path': image_path,
                        'width': width,
                        'height': height,
                        'img_id': img_id,
                        'instances': [{
                            'bbox': box,
                            'bbox_label': bbox_label,
                            'ignore_flag': 0,
                        }],
                        'text': text
                    })

                count += 1

        if not data_list:
            raise ValueError(
                f'No samples found in split file(s): {self.split_file}')

        return data_list