import os
import os.path as osp
from PIL import Image
from typing import Dict, List, Union
import xml.etree.ElementTree as ET
import numpy as np
import mmengine
from mmengine.dataset import BaseDataset


class DIORRRSVGDataset(BaseDataset):
    """DIOR-R-RS-VG. Visual Grounding in Remote Sensing dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``.
        ann_file (str): Annotation file path.
        split_file (str): Split file path.
        data_prefix (str): Prefix for training data.
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
        """Load DIOR-R-RSVG annotation list, compatible with RandomSamplingNegPos transform"""
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
            with Image.open(image_path) as img:
                width, height = img.size

            # 1. 先收集当前图片所有合法目标
            all_targets = []
            for member in root.findall('object'):
                if count not in indices:
                    count += 1
                    continue
                bnd_box = member.find('robndbox')
                if bnd_box is None:
                    count += 1
                    continue
                polygon = np.array([
                    float(bnd_box.find('x_left_top').text),
                    float(bnd_box.find('y_left_top').text),
                    float(bnd_box.find('x_right_top').text),
                    float(bnd_box.find('y_right_top').text),
                    float(bnd_box.find('x_right_bottom').text),
                    float(bnd_box.find('y_right_bottom').text),
                    float(bnd_box.find('x_left_bottom').text),
                    float(bnd_box.find('y_left_bottom').text),
                ]).astype(np.float32)
                description = member.findtext('description', default='')
                if len(description) <= 1:
                    count += 1
                    continue
                all_targets.append({
                    'polygon': polygon,
                    'phrase': description.strip()
                })
                count += 1

            # 当前图片无有效目标，跳过
            if len(all_targets) == 0:
                continue

            # 2. 拼接全局文本 + 计算每个phrase的token位置偏移（模拟tokens_positive）
            global_text_parts = []
            phrase_offsets = []  # 记录每个phrase在全文中的起始字符
            cur_offset = 0
            for t in all_targets:
                phrase = t['phrase']
                global_text_parts.append(phrase)
                phrase_offsets.append(cur_offset)
                # 加分隔符，保证句子分割
                cur_offset += len(phrase) + 2  # 句子后加 . 空格

            full_text = '. '.join(global_text_parts) + '.'
            instances = []
            phrases_dict = {}

            # 3. 遍历每个目标，构造instances和phrases
            for idx, target in enumerate(all_targets):
                # 实例信息
                instance = {
                    'bbox': target['polygon'],  # 遥感旋转四边形，模型侧自行处理
                    'bbox_label': idx,  # 和原VG逻辑对齐：label=region索引
                    'ignore_flag': 0
                }
                instances.append(instance)

                # 构造tokens_positive：这里先用字符偏移，tokenizer会在transform里转token下标
                start_char = phrase_offsets[idx]
                end_char = start_char + len(target['phrase'])
                phrases_dict[idx] = {
                    'phrase': target['phrase'],
                    'tokens_positive': [[start_char, end_char]]
                }

            # 4. 组装最终data_info，字段完全对齐ODVGDataset
            data_info = {
                'img_path': image_path,
                'width': width,
                'height': height,
                'img_id': img_id,
                'instances': instances,
                'text': full_text,
                'phrases': phrases_dict,
                'dataset_mode': 'VG'  # 标记为VG模式，和ODVGDataset统一
            }
            data_list.append(data_info)

        if not data_list:
            raise ValueError(
                f'No samples found in split file(s): {self.split_file}')
        return data_list


class DIORRRSVGValDataset(DIORRRSVGDataset):

    def load_data_list(self) -> List[dict]:
        """Load DIOR-RSVG annotation list."""
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
                    bnd_box = member.find('robndbox')
                    if bnd_box is None:
                        count += 1
                        continue

                    polygon = np.array([
                        float(bnd_box.find('x_left_top').text),
                        float(bnd_box.find('y_left_top').text),
                        float(bnd_box.find('x_right_top').text),
                        float(bnd_box.find('y_right_top').text),
                        float(bnd_box.find('x_right_bottom').text),
                        float(bnd_box.find('y_right_bottom').text),
                        float(bnd_box.find('x_left_bottom').text),
                        float(bnd_box.find('y_left_bottom').text),
                    ]).astype(np.float32)

                    description = member.findtext('description', default='')
                    if len(description) <= 1:
                        count += 1
                        continue

                    data_list.append({
                        'img_path': image_path,
                        'width': width,
                        'height': height,
                        'img_id': img_id,
                        'instances': [{
                            'bbox': polygon,
                            'ignore_flag': 0,
                        }],
                        'text': description,
                        'custom_entities': False,
                        'tokens_positive': -1
                    })

                count += 1

        if not data_list:
            raise ValueError(
                f'No samples found in split file(s): {self.split_file}')

        return data_list