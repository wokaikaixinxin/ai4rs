import os
import os.path as osp
import xml.etree.ElementTree as ET
from typing import Dict, List, Union
import numpy as np
from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DIORRRSVGDetDataset(BaseDataset):

    METAINFO = {
        'classes':
        ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
         'chimney', 'expressway-service-area', 'expressway-toll-station',
         'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
         'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
         'windmill'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42),
                    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0),
                    (120, 166, 157)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'Annotations_obb',
                 split_file: Union[str, list] = 'train.txt',
                 data_prefix: Dict = dict(img_path='JPEGImages/'),
                 **kwargs):
        self.split_file = split_file
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs)

    def _join_prefix(self):
        if isinstance(self.split_file, str):
            split_file_list = [osp.join(self.data_root, self.split_file)]
        else:
            split_file_list = [
                osp.join(self.data_root, file_name)
                for file_name in self.split_file
            ]

        self.split_file = split_file_list
        return super()._join_prefix()

    def load_data_list(self) -> List[dict]:
        data_list = []

        anno_files = sorted(
            osp.join(self.ann_file, file_name)
            for file_name in os.listdir(self.ann_file)
            if file_name.endswith('.xml'))

        indices = set()
        for split_path in self.split_file:
            with open(split_path, 'r') as f:
                indices.update(int(line.strip()) for line in f if line.strip())

        class_to_label = {
            class_name: i
            for i, class_name in enumerate(self.metainfo['classes'])
        }

        img_prefix = self.data_prefix['img_path']
        count = 0

        for anno_path in anno_files:
            root = ET.parse(anno_path).getroot()

            filename = root.findtext('filename')
            if not filename:
                continue

            image_path = osp.join(img_prefix, filename)
            img_id = osp.splitext(filename)[0]

            size = root.find('size')
            if size is None:
                continue

            width = int(size.findtext('width'))
            height = int(size.findtext('height'))

            instances = []

            for member in root.findall('object'):
                current_index = count
                count += 1

                if current_index not in indices:
                    continue

                cls_name = member.find('name').text.lower()
                if cls_name not in class_to_label:
                    continue

                robndbox = member.find('robndbox')
                if robndbox is None:
                    continue

                polygon = np.array([
                    float(robndbox.findtext('x_left_top')),
                    float(robndbox.findtext('y_left_top')),
                    float(robndbox.findtext('x_right_top')),
                    float(robndbox.findtext('y_right_top')),
                    float(robndbox.findtext('x_right_bottom')),
                    float(robndbox.findtext('y_right_bottom')),
                    float(robndbox.findtext('x_left_bottom')),
                    float(robndbox.findtext('y_left_bottom')),
                ], dtype=np.float32)

                instances.append({
                    'bbox': polygon,
                    'bbox_label': class_to_label[cls_name],
                    'ignore_flag': 0
                })

            if not instances:
                continue

            data_list.append({
                'img_path': image_path,
                'img_id': img_id,
                'width': width,
                'height': height,
                'instances': instances
            })

        if not data_list:
            raise ValueError(
                f'No samples found in split file(s): {self.split_file}')

        return data_list