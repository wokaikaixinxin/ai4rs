import json
import os.path as osp
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class VRSBenchDetDataset(BaseDataset):
    """VRSBench rotated object detection dataset.

    Every JSON file describes one image. ``obj_corner`` stores a normalized
    quadrilateral in ``(x1, y1, ..., x4, y4)`` order, which is converted to
    image-space coordinates before it is passed to the detection pipeline.
    """

    METAINFO = {
        'classes':
            ('airplane', 'airport', 'baseball-diamond', 'basketball-court',
             'bridge','chimney', 'container-crane', 'dam','expressway-service-area',
             'expressway-toll-station', 'golffield', 'ground-track-field', 'harbor',
             'helicopter', 'helipad', 'overpass', 'roundabout', 'ship',
             'soccer-ball-field', 'stadium', 'storage-tank', 'swimming-pool',
             'tennis-court', 'trainstation', 'vehicle', 'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 128, 0),
            (255, 0, 128), (128, 255, 0), (0, 255, 128), (0, 128, 255), (128, 0, 255),
            (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128), (255, 128, 255),
            (128, 255, 255)]
    }

    def __init__(self,
                 data_root: str = '',
                 ann_file: str = 'Annotations_train',
                 data_prefix: dict = dict(img_path='Images_train'),
                 **kwargs) -> None:

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load all image-level JSON annotations from ``ann_file``."""
        annotation_dir = Path(self.ann_file)
        if not annotation_dir.is_dir():
            raise FileNotFoundError(
                f'VRSBench annotation directory does not exist: '
                f'{annotation_dir}')

        classes = tuple(self.metainfo.get('classes', ()))
        if not classes:
            raise ValueError('VRSBenchDetDataset requires non-empty classes.')
        cat2label: Dict[str, int] = {name: idx for idx, name in enumerate(classes)}

        data_list = []
        for ann_path in sorted(annotation_dir.glob('*.json')):
            with ann_path.open('r', encoding='utf-8') as file:
                annotation = json.load(file)

            image_name = annotation.get('image')
            if not image_name:
                raise ValueError(f'Missing image name in {ann_path}.')

            img_path = osp.join(self.data_prefix['img_path'], image_name)
            if not osp.isfile(img_path):
                raise FileNotFoundError(
                    f'VRSBench image does not exist: {img_path}')
            with Image.open(img_path) as image:
                width, height = image.size

            instances = []
            for obj_index, obj in enumerate(annotation.get('objects', [])):
                category = obj.get('obj_cls')
                if not isinstance(category, str):
                    raise ValueError(
                        f'Missing obj_cls in {ann_path}, object {obj_index}.')
                category = category.strip().lower()
                if category not in cat2label:
                    raise ValueError(
                        f'Unknown VRSBench category {category!r} in '
                        f'{ann_path}, object {obj_index}.')

                qbox = np.asarray(obj.get('obj_corner'), dtype=np.float32)
                if qbox.shape != (8, ):
                    raise ValueError(
                        f'obj_corner must have shape (8,) in {ann_path}, '
                        f'object {obj_index}, but got {qbox.shape}.')
                qbox[0::2] *= width
                qbox[1::2] *= height

                instances.append({
                    'bbox': qbox,
                    'bbox_label': cat2label[category],
                    'ignore_flag': 0,
                })

            if not instances:
                continue
            data_list.append({
                'img_id': osp.splitext(image_name)[0],
                'img_path': img_path,
                'height': height,
                'width': width,
                'instances': instances,
            })

        if not data_list:
            raise ValueError(
                f'No valid samples found in annotation directory: '
                f'{annotation_dir}')
        return data_list