import os.path as osp
from pathlib import Path
import json
from PIL import Image
from typing import List
import numpy as np
from mmengine.dataset import BaseDataset


class VRSBenchVGDataset(BaseDataset):
    """VRSBench visual grounding dataset.

    Training annotations are stored as one json file per image. Each object in
    the json contributes one referring expression and one oriented box. The
    returned fields follow the VG branch of ``ODVGDataset.load_data_list()``:
    ``img_path``, ``height``, ``width``, ``text``, ``instances``, ``phrases``
    and ``dataset_mode='VG'``.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'Annotations_train',
                 data_prefix: dict = dict(img_path='Images_train'),
                 **kwargs):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        ann_files = sorted([str(p) for p in Path(self.ann_file).glob("*.json")])

        data_list = []
        for ann_path in ann_files:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)

            filename = ann.get('image', '')
            if not filename:
                filename = osp.splitext(osp.basename(ann_path))[0] + '.png'
            img_path = osp.join(self.data_prefix['img_path'], filename)
            if not osp.exists(img_path):
                continue
            with Image.open(img_path) as img:
                width, height = img.size

            text = ''
            instances = []
            phrase_map = {}
            for idx, obj in enumerate(ann.get('objects', [])):
                phrase = obj.get('referring_sentence', '').strip().rstrip('.')
                if len(phrase) <= 1:
                    continue
                qbox = obj.get('obj_corner')
                if qbox is None or len(qbox) != 8:
                    continue
                qbox = np.asarray(qbox, dtype=np.float32)
                qbox[0::2] *= width
                qbox[1::2] *= height

                if idx > 0:
                    text += '. '
                start = len(text)
                text += phrase
                end = len(text)
                instances.append({
                    'bbox': qbox,
                    'bbox_label': idx,
                    'ignore_flag': 0
                })
                phrase_map[idx] = {
                    'phrase': phrase,
                    'tokens_positive': [[start, end]]
                }
            text += '.'
            if not instances:
                continue

            img_id = osp.splitext(filename)[0]
            data_list.append({
                'img_path': img_path,
                'height': height,
                'width': width,
                'img_id': img_id,
                'text': text,
                'instances': instances,
                'phrases': phrase_map,
                'dataset_mode': 'VG'
            })

        if not data_list:
            raise ValueError(f'No samples found in annotation dir: '
                             f'{self.ann_file}')
        return data_list


class VRSBenchVGValDataset(VRSBenchVGDataset):
    """VRSBench referring-expression validation split.

    ``VRSBench_EVAL_referring.json`` is a list of referring expressions. Each
    expression is returned as an individual VG sample.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'VRSBench_EVAL_referring.json',
                 data_prefix: dict = dict(img_path='Images_val'),
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        data_list = []
        for ann in annotations:
            filename = ann.get('image_id', ann.get('image'))
            if not filename:
                continue

            text = ann.get('question', '').strip()
            if len(text) <= 1:
                continue

            img_path = osp.join(self.data_prefix['img_path'], filename)
            if not osp.exists(img_path):
                continue
            with Image.open(img_path) as img:
                width, height = img.size
            qbox = ann.get('obj_corner')
            if qbox is None or len(qbox) != 8:
                continue
            qboxes = [np.asarray(qbox, dtype=np.float32)]
            qboxes[0][0::2] *= width
            qboxes[0][1::2] *= height

            img_id = osp.splitext(filename)[0]

            instances = [{'bbox': qbox, 'ignore_flag': 0} for qbox in qboxes]

            data_list.append({
                'img_path': img_path,
                'height': height,
                'width': width,
                'img_id': img_id,
                'text': text,
                'instances': instances,
                'custom_entities': False,
                'tokens_positive': -1,
                'dataset_mode': 'VG'
            })

        if not data_list:
            raise ValueError(f'No samples found in annotation file: '
                             f'{self.ann_file}')
        return data_list