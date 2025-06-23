# Copyright (c) ai4rs. All rights reserved.
import glob
import os.path as osp
from typing import List
from mmengine.dataset import BaseDataset
from ai4rs.registry import DATASETS

@DATASETS.register_module()
class ICDAR15Dataset(BaseDataset):
    '''ICDAR2015 dataset for detection.
        Args:
            img_suffix (str): The suffix of images. Defaults to 'jpg'.
            filter_difficulty (bool):
    '''
    METAINFO = {
        'classes': ('text',),    # Note: The comma ',' after 'text' cannot be omitted !!!
        'palette': [(255, 0, 0)] # palette is a list of color tuples, which is used for visualization.
    }

    def __init__(self,
                 img_suffix: str = 'jpg',
                 filter_difficulty: bool = True,
                 **kwargs) -> None:
        self.img_suffix = img_suffix
        self.filter_difficulty = filter_difficulty
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            'classes in `ICDAR15Dataset` can not be None.'
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }
        data_list = []
        if self.ann_file == '': # for test
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
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
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in '
                                 f'{self.ann_file}')
            for txt_file in txt_files:
                print(txt_file)
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id[3:] + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file, encoding='UTF-8-sig') as f:
                    s = f.readlines()
                    for si in s:
                        si = si.rstrip('\n')
                        instance = {}
                        bbox_info = si.split(',')
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = 'text'
                        instance['bbox_label'] = cls_map[cls_name]
                        instance['ignore_flag'] = 0
                        text = bbox_info[8]
                        if self.filter_difficulty and text == '###':
                            continue
                        else:
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
        """Get ICDAR2015 category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]