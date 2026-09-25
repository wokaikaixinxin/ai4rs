import json
import os.path as osp
from typing import Dict, List
import numpy as np
from PIL import Image
from mmengine.dataset import BaseDataset


class AVVGDetDataset(BaseDataset):
    """AVVG detection dataset with quadrilateral annotations.

    Each JSONL record contains an image name, a text category in ``question``
    and one or more quadrilaterals in ``poly``. Records are grouped by image so
    that one dataset item contains all annotations belonging to that image.
    """

    METAINFO = {
        'classes':
            ('aito car', 'aion car', 'audi car', 'baojun car', 'beijing car',
             'black car', 'blue car', 'bmw car', 'brown car', 'buick car',
             'byd car', 'cadillac car', 'changan car', 'chery car', 'chevrolet car',
             'citroen car', 'compact car', 'crossover car', 'deepal car', 'denza car',
             'dfsk car', 'ford car', 'full-size car', 'geely car', 'golden car',
             'gray car', 'green car', 'harvard car', 'haval car', 'honda car',
             'hongqi car', 'hyundai car', 'ideal car', 'infiniti car', 'jaguar car',
             'jeep car', 'kia car', 'leap car', 'lexus car', 'lincoln car',
             'lynk&co car', 'mazda car', 'mercedes-benz car', 'mg car', 'micro-size car',
             'mid-size car', 'mini car', 'mitsubishi car', 'mpv car', 'neta car',
             'nio car', 'nissan car', 'ora car', 'orange car', 'peugeot car',
             'pink car', 'porsche car', 'purple car', 'red car', 'renault car',
             'riich car', 'rising car', 'roewe car', 'sedan car', 'silver car',
             'skoda car', 'smart car', 'sports car', 'subcompact car', 'suv car',
             'tesla car', 'toyota car', 'vgv car', 'volkswagen car', 'volvo car',
             'wey car', 'white car', 'wuling car', 'xpeng car', 'yellow car'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 128, 0),
            (255, 0, 128), (128, 255, 0), (0, 255, 128), (0, 128, 255), (128, 0, 255),
            (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128), (255, 128, 255),
            (128, 255, 255), (255, 200, 0), (255, 150, 0), (200, 255, 0), (150, 255, 0),
            (0, 255, 200), (0, 200, 255), (0, 150, 255), (150, 0, 255), (200, 0, 255),
            (255, 0, 200), (100, 100, 100), (180, 180, 180), (220, 220, 220), (255, 255, 255),
            (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 100), (200, 255, 100),
            (100, 200, 255), (200, 100, 255), (255, 100, 200), (100, 255, 200), (255, 180, 180),
            (180, 255, 180), (180, 180, 255), (255, 255, 180), (255, 180, 255), (180, 255, 255),
            (255, 160, 0), (160, 255, 0), (0, 255, 160), (0, 160, 255), (160, 0, 255),
            (255, 0, 160), (255, 240, 0), (0, 255, 240), (240, 0, 255), (255, 0, 240),
            (240, 255, 0), (0, 240, 255), (255, 220, 150), (220, 255, 150), (150, 220, 255),
            (220, 150, 255), (255, 150, 220), (150, 255, 220), (200, 200, 200), (120, 120, 120),
            (80, 80, 80), (60, 60, 60), (40, 40, 40), (20, 20, 20), (0, 0, 0)]

    }

    def __init__(self,
                 data_root: str = '',
                 ann_file: str = 'metainfo/avvg_detection_train.jsonl',
                 data_prefix: dict = dict(img_path='images/avvg'),
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load image-level detection annotations from a JSONL file."""
        classes = tuple(self.metainfo.get('classes', ()))
        if not classes:
            raise ValueError('AVVGDetDataset requires non-empty classes.')
        cat2label = {
            ' '.join(name.strip().lower().split()): idx
            for idx, name in enumerate(classes)
        }

        image_records: Dict[str, List[dict]] = {}
        seen = set()
        with open(self.ann_file, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'Invalid JSON in {self.ann_file}:{line_number}.') from exc

                image_id = str(record.get('image_id', '')).strip()
                question = record.get('question', '')
                if not image_id or not isinstance(question, str):
                    continue
                question = ' '.join(
                    question.strip().lower().rstrip('.').split())
                if not question:
                    continue
                if question not in cat2label:
                    raise ValueError(
                        f'Unknown AVVG category {question!r} in '
                        f'{self.ann_file}:{line_number}.')

                polygons = np.asarray(record['poly'],
                                      dtype=np.float32).reshape(-1, 8)

                label = cat2label[question]
                records = image_records.setdefault(image_id, [])
                for polygon in polygons:
                    polygon = polygon.astype(np.float32, copy=False)
                    key = (image_id, label, tuple(polygon.tolist()))
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append({
                        'bbox': polygon,
                        'bbox_label': label,
                        'ignore_flag': 0,
                    })

        data_list = []
        for image_id, instances in image_records.items():
            img_path = osp.join(self.data_prefix['img_path'], image_id)
            if not osp.isfile(img_path):
                raise FileNotFoundError(
                    f'AVVG image does not exist: {img_path}')

            with Image.open(img_path) as image:
                width, height = image.size

            if not instances:
                continue
            data_list.append({
                'img_path': img_path,
                'img_id': osp.splitext(osp.basename(image_id))[0],
                'width': width,
                'height': height,
                'instances': instances,
            })

        if not data_list:
            raise ValueError(
                f'No valid samples found in annotation file: {self.ann_file}')
        return data_list

class AVVGDetOneClassDataset(BaseDataset):
    METAINFO = {
        'classes':
        ('object'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60)]
    }
    def load_data_list(self) -> List[dict]:
        """Load image-level detection annotations from a JSONL file."""
        classes = tuple(self.metainfo.get('classes', ()))
        if not classes:
            raise ValueError('AVVGDetDataset requires non-empty classes.')

        image_records: Dict[str, List[dict]] = {}
        seen = set()
        with open(self.ann_file, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'Invalid JSON in {self.ann_file}:{line_number}.') from exc

                image_id = str(record.get('image_id', '')).strip()
                polygons = np.asarray(record['poly'],
                                      dtype=np.float32).reshape(-1, 8)
                label = 0
                records = image_records.setdefault(image_id, [])
                for polygon in polygons:
                    polygon = polygon.astype(np.float32, copy=False)
                    key = (image_id, label, tuple(polygon.tolist()))
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append({
                        'bbox': polygon,
                        'bbox_label': label,
                        'ignore_flag': 0,
                    })

        data_list = []
        for image_id, instances in image_records.items():
            img_path = osp.join(self.data_prefix['img_path'], image_id)
            if not osp.isfile(img_path):
                raise FileNotFoundError(
                    f'AVVG image does not exist: {img_path}')

            with Image.open(img_path) as image:
                width, height = image.size

            if not instances:
                continue
            data_list.append({
                'img_path': img_path,
                'img_id': osp.splitext(osp.basename(image_id))[0],
                'width': width,
                'height': height,
                'instances': instances,
            })

        if not data_list:
            raise ValueError(
                f'No valid samples found in annotation file: {self.ann_file}')
        return data_list