# Copyright (c) ai4rs. All rights reserved.
import glob
import os.path as osp
from typing import List
from mmengine.dataset import BaseDataset
from ai4rs.registry import DATASETS

@DATASETS.register_module()
class RSARDataset(BaseDataset):
    """RSAR dataset for detection.

    Note: ``ann_file`` in RSARDataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In RSARDataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    METAINFO = {
        'classes': ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (0, 0, 230), (106, 0, 228),
         (0, 182, 0), (200, 182, 0), (0, 182, 200)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], '*'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            img_files = glob.glob(osp.join(self.data_prefix['img_path'], '*'))
            if len(img_files) == 0:
                raise ValueError('There is no img file in '
                                 f'{self.data_prefix["img_path"]}')
            for img_file in img_files:
                data_info = {}
                img_id = osp.split(img_file)[1][:-4]
                data_info['img_id'] = img_id
                data_info['file_name'] = osp.basename(img_file)
                data_info['img_path'] = img_file
                instances = []
                txt_file = osp.join(self.ann_file, img_id + ".txt")
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get RSAR category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]