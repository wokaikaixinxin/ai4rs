import json
import os.path as osp
from typing import List

import numpy as np
from PIL import Image
from mmengine.dataset import BaseDataset


class AVVGDataset(BaseDataset):
    """AVVG rotated visual grounding dataset."""

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'metainfo/avvg_train.jsonl',
                 data_prefix: dict = dict(img_path='images/avvg'),
                 **kwargs):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        seen = set()
        image_to_records = {}

        with open(self.ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)

                image_id = record.get('image_id', '')
                question = record.get('question', '').strip()
                poly = record.get('poly')
                if not image_id or len(question) <= 1 or poly is None:
                    continue

                if isinstance(poly, list) and poly and isinstance(
                        poly[0], (list, tuple)):
                    poly = [coord for point in poly for coord in point]

                qbox = np.asarray(poly, dtype=np.float32).reshape(-1)
                if qbox.size != 8:
                    continue

                sample_key = (image_id, question, tuple(qbox.tolist()))
                if sample_key in seen:
                    continue
                seen.add(sample_key)

                image_to_records.setdefault(image_id, []).append({
                    'question': question.rstrip('.'),
                    'qbox': qbox
                })

        data_list = []
        for image_id, records in image_to_records.items():
            img_path = osp.join(self.data_prefix['img_path'], image_id)
            if not osp.exists(img_path):
                continue

            with Image.open(img_path) as img:
                width, height = img.size

            text = ''
            instances = []
            phrase_map = {}
            tokens_positive = []
            for idx, record in enumerate(records):
                question = record['question']
                qbox = record['qbox']

                if text:
                    text += '. '
                start = len(text)
                text += question
                end = len(text)

                instances.append({
                    'bbox': qbox,
                    'bbox_label': idx,
                    'ignore_flag': 0
                })
                phrase_map[idx] = {
                    'phrase': question,
                    'tokens_positive': [[start, end]]
                }
                tokens_positive.append([[start, end]])
            text += '.'
            if not instances:
                continue

            img_id = osp.splitext(image_id)[0]
            data_list.append({
                'img_path': img_path,
                'width': width,
                'height': height,
                'img_id': img_id,
                'text': text,
                'instances': instances,
                'phrases': phrase_map,
                'custom_entities': False,
                'tokens_positive': tokens_positive,
                'dataset_mode': 'VG'
            })

        if not data_list:
            raise ValueError(f'No samples found in annotation file: '
                             f'{self.ann_file}')
        return data_list


class AVVGValDataset(AVVGDataset):
    """AVVG validation split with one referring expression per sample."""

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'metainfo/avvg_test.jsonl',
                 data_prefix: dict = dict(img_path='images/avvg'),
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            data_prefix=data_prefix,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        seen = set()

        with open(self.ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                image_id = record.get('image_id', '')
                question = record.get('question', '').strip()
                poly = record.get('poly')
                if not image_id or len(question) <= 1 or poly is None:
                    continue

                if isinstance(poly, list) and poly and isinstance(
                        poly[0], (list, tuple)):
                    poly = [coord for point in poly for coord in point]
                qbox = np.asarray(poly, dtype=np.float32).reshape(-1)
                if qbox.size != 8:
                    continue

                sample_key = (image_id, question, tuple(qbox.tolist()))
                if sample_key in seen:
                    continue
                seen.add(sample_key)

                img_path = osp.join(self.data_prefix['img_path'], image_id)
                if not osp.exists(img_path):
                    continue
                with Image.open(img_path) as img:
                    width, height = img.size

                data_list.append({
                    'img_path': img_path,
                    'width': width,
                    'height': height,
                    'img_id': osp.splitext(image_id)[0],
                    'instances': [{
                        'bbox': qbox,
                        'ignore_flag': 0,
                    }],
                    'text': question,
                    'custom_entities': False,
                    'tokens_positive': -1,
                    'dataset_mode': 'VG'
                })

        if not data_list:
            raise ValueError(f'No samples found in annotation file: '
                             f'{self.ann_file}')
        return data_list