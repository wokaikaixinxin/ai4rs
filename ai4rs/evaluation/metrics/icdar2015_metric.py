# Copyright (c) ai4rs. All rights reserved.
import os
import os.path as osp
import torch
import tempfile
import numpy as np
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union
from torch import Tensor

from mmengine.logging import MMLogger
from ai4rs.evaluation import eval_rbbox_map
from ai4rs.registry import METRICS
from ai4rs.evaluation.metrics import DOTAMetric

@METRICS.register_module()
class ICDAR2015Metric(DOTAMetric):
    default_prefix: Optional[str] = 'icdar2015'
    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 merge_patches: bool = False,
                 iou_thr: float = 0.1,
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(
            iou_thrs=iou_thrs,
            scale_ranges=scale_ranges,
            metric=metric,
            predict_box_type=predict_box_type,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            merge_patches=merge_patches,
            iou_thr=iou_thr,
            eval_mode=eval_mode,
            collect_device=collect_device,
            prefix=prefix)

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            # convert predictions to txt format and dump to zip file
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 5)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError
        return eval_results


    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for ICDAR2015 offline or online evaluation.

        res_img_1.txt, res_img_2.txt, ..., res_img_500.txt ----> submit.zip

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        logger: MMLogger = MMLogger.get_current_instance()
        collector = defaultdict(list)

        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            oriname = img_id
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            label_dets = np.concatenate(
                [labels[:, np.newaxis], bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            id_list.append(oriname)
            dets_list.append(label_dets_list)

        outfile_prefix_new = ''
        for i in range(1, 999):
            outfile_prefix_new = outfile_prefix[:-1] + f'_{i}/'
            if not osp.exists(outfile_prefix_new):
                os.makedirs(outfile_prefix_new)
                break
        if outfile_prefix_new == '':
            raise FileExistsError

        files = [
            osp.join(outfile_prefix_new, 'res_' + idx + '.txt')
            for idx in id_list
        ]
        file_objs = [open(f, 'w') for f in files]

        for img_id, dets_per_img, f in zip(id_list, dets_list, file_objs):
            logger.info(f'saving {f}')
            for dets in dets_per_img:
                if dets.size == 0:
                    continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    labels, rboxes, scores = torch.split(th_dets, (1, 5, 1), dim=-1)
                    qboxes = self.rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    labels, qboxes, scores = torch.split(th_dets, (1, 8, 1), dim=-1)
                else:
                    raise NotImplementedError
                for qbox, score in zip(qboxes, scores):
                    txt_element = [f'{int(p)}' for p in qbox] + \
                                  [str(round(float(score), 4))]
                    f.writelines(','.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        zip_path = osp.join(outfile_prefix_new, 'submit.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return zip_path

    def rbox2qbox(self, boxes: Tensor) -> Tensor:
        """Convert rotated boxes to quadrilateral boxes.

        Args:
            boxes (Tensor): Rotated box tensor with shape of (..., 5).

        Returns:
            Tensor: Quadrilateral box tensor with shape of (..., 8).
        """
        ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
        cos_value, sin_value = torch.cos(theta), torch.sin(theta)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        pt1 = ctr - vec1 - vec2 # left bottom
        pt2 = ctr + vec1 - vec2 # right bottom
        pt3 = ctr + vec1 + vec2 # right top
        pt4 = ctr - vec1 + vec2 # left top
        return torch.cat([pt1, pt2, pt3, pt4], dim=-1)

