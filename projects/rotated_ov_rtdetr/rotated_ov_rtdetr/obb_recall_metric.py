from terminaltables import AsciiTable
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from mmcv.ops import box_iou_quadri
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import qbox2rbox, rbbox_overlaps


def _recalls(all_ious, proposal_nums, thrs):

    img_num = len(all_ious)
    total_gt_num = sum([ious.shape[0] for ious in all_ious])
    if total_gt_num == 0:
        return np.zeros((proposal_nums.size, thrs.size), dtype=np.float32)

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None,
                         logger=None):
    """Print recalls in a table.

    Args:
        recalls (ndarray): calculated from `bbox_recalls`
        proposal_nums (ndarray or list): top N proposals
        iou_thrs (ndarray or list): iou thresholds
        row_idxs (ndarray): which rows(proposal nums) to print
        col_idxs (ndarray): which cols(iou thresholds) to print
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmengine.logging.print_log()` for details.
            Default: None.
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format."""
    if proposal_nums is None:
        proposal_nums = (100, 300, 1000)

    if isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, Sequence):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs


def _empty_boxes(box_dim: int) -> np.ndarray:
    """Create an empty box array with a stable shape."""
    return np.zeros((0, box_dim), dtype=np.float32)


def _to_numpy(data) -> np.ndarray:
    """Convert tensors/BaseBoxes/list-like data to a numpy array."""
    if data is None:
        return np.zeros((0, ), dtype=np.float32)
    if hasattr(data, 'tensor'):
        data = data.tensor
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = np.asarray(data, dtype=np.float32)
    return data


def _normalize_box_array(boxes,
                         empty_dim: int = 5,
                         with_score: bool = False) -> np.ndarray:
    """Normalize boxes to a 2-D numpy array.

    Supported box layouts are rbox (5), qbox (8), and the same layouts with a
    trailing score column (6/9).
    """
    boxes = _to_numpy(boxes)
    valid_dims = (6, 9) if with_score else (5, 8)
    if boxes.size == 0:
        if boxes.ndim == 2 and boxes.shape[1] in valid_dims:
            return boxes.astype(np.float32, copy=False)
        return _empty_boxes(empty_dim + int(with_score))
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2:
        raise ValueError(f'boxes should be a 2-D array, but got {boxes.shape}')

    if boxes.shape[1] not in valid_dims:
        kind = 'scored boxes' if with_score else 'boxes'
        raise ValueError(
            f'{kind} should have shape (N, {valid_dims}), '
            f'but got {boxes.shape}')
    return boxes.astype(np.float32, copy=False)


def _split_proposals(proposals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split proposal boxes and optional scores, sorting by score if present."""
    proposals = _to_numpy(proposals)
    if proposals.size == 0:
        if proposals.ndim == 2 and proposals.shape[1] in (5, 8):
            return proposals.astype(np.float32, copy=False), np.zeros(
                (0, ), dtype=np.float32)
        if proposals.ndim == 2 and proposals.shape[1] in (6, 9):
            return proposals[:, :-1].astype(
                np.float32, copy=False), np.zeros((0, ), dtype=np.float32)
        return _empty_boxes(5), np.zeros((0, ), dtype=np.float32)
    if proposals.ndim == 1:
        proposals = proposals.reshape(1, -1)
    if proposals.ndim != 2 or proposals.shape[1] not in (5, 6, 8, 9):
        raise ValueError(
            'proposals should have rbox/qbox format with optional scores, '
            f'but got {proposals.shape}')

    if proposals.shape[1] in (6, 9):
        scores = proposals[:, -1]
        boxes = proposals[:, :-1]
        sort_idx = np.argsort(scores)[::-1]
        return boxes[sort_idx].astype(np.float32, copy=False), scores[sort_idx]

    return proposals.astype(np.float32, copy=False), np.zeros(
        (proposals.shape[0], ), dtype=np.float32)


def _as_rboxes(boxes: np.ndarray) -> np.ndarray:
    """Convert rbox/qbox arrays to rbox arrays."""
    if boxes.size == 0:
        return _empty_boxes(5)
    if boxes.shape[1] == 5:
        return boxes.astype(np.float32, copy=False)
    if boxes.shape[1] == 8:
        return qbox2rbox(torch.from_numpy(boxes).float()).cpu().numpy()
    raise ValueError(f'Only rbox/qbox are supported, but got {boxes.shape}')


def obb_overlaps(gts: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    """Calculate IoU matrix between GT OBBs and proposal OBBs.

    Args:
        gts (ndarray): GT boxes in rbox (N, 5) or qbox (N, 8) format.
        proposals (ndarray): Proposal boxes in rbox (K, 5) or qbox (K, 8)
            format.

    Returns:
        ndarray: IoU matrix with shape (N, K).
    """
    if gts.size == 0:
        return np.zeros((0, proposals.shape[0]), dtype=np.float32)
    if proposals.size == 0:
        return np.zeros((gts.shape[0], 0), dtype=np.float32)

    if gts.shape[1] == 8 and proposals.shape[1] == 8:
        ious = box_iou_quadri(
            torch.from_numpy(gts).float(),
            torch.from_numpy(proposals).float())
    else:
        ious = rbbox_overlaps(
            torch.from_numpy(_as_rboxes(gts)).float(),
            torch.from_numpy(_as_rboxes(proposals)).float())
    return ious.cpu().numpy().astype(np.float32, copy=False)


def eval_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 iou_thrs=0.5,
                 logger=None):
    """Calculate recalls for oriented bounding boxes.

    Args:
        gts (list[ndarray]): Arrays of shape (N, 5) for rboxes or (N, 8) for
            qboxes.
        proposals (list[ndarray]): Arrays of shape (K, 5)/(K, 8), or with a
            trailing score column: (K, 6)/(K, 9).
        proposal_nums (int | Sequence[int]): Top N proposals to be evaluated.
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmengine.logging.print_log()` for details.
            Default: None.

    Returns:
        ndarray: recalls of different ious and proposal nums
    """

    img_num = len(gts)
    assert img_num == len(proposals)
    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)
    all_ious = []
    for i in range(img_num):
        gt = _normalize_box_array(gts[i])
        img_proposal, _ = _split_proposals(proposals[i])
        prop_num = min(img_proposal.shape[0], proposal_nums.max())
        if gt.shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = obb_overlaps(gt, img_proposal[:prop_num])
        all_ious.append(ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)

    print_recall_summary(recalls, proposal_nums, iou_thrs, logger=logger)
    return recalls


@METRICS.register_module()
class OBBRecallMetric(BaseMetric):
    """Fast proposal recall metric for DIOR-R oriented bounding boxes."""

    default_prefix: Optional[str] = 'obbrecall'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'proposal_fast',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    f"metric should be one of 'proposal_fast', but got {metric}.")

        self.proposal_nums = list(proposal_nums)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        if self.iou_thrs is None:
            self.iou_thrs = [0.5]
        self.metric_items = metric_items
        self.classwise = classwise
        self.format_only = format_only
        self.outfile_prefix = outfile_prefix
        self.sort_categories = sort_categories
        self.use_mp_eval = use_mp_eval

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Collect GT and predictions from DIOR-R data samples."""
        for data_sample in data_samples:
            gt_instances = data_sample.get('gt_instances', {})
            ignored_instances = data_sample.get('ignored_instances', {})

            if gt_instances == {}:
                ann = dict(
                    bboxes=_empty_boxes(5),
                    labels=np.zeros((0, ), dtype=np.int64),
                    bboxes_ignore=_empty_boxes(5),
                    labels_ignore=np.zeros((0, ), dtype=np.int64))
            else:
                bboxes = _normalize_box_array(gt_instances['bboxes'])
                labels = _to_numpy(gt_instances['labels']).astype(np.int64)
                if ignored_instances == {}:
                    bboxes_ignore = _empty_boxes(bboxes.shape[1])
                    labels_ignore = np.zeros((0, ), dtype=np.int64)
                else:
                    bboxes_ignore = _normalize_box_array(
                        ignored_instances['bboxes'],
                        empty_dim=bboxes.shape[1])
                    labels_ignore = _to_numpy(
                        ignored_instances['labels']).astype(np.int64)
                ann = dict(
                    bboxes=bboxes,
                    labels=labels,
                    bboxes_ignore=bboxes_ignore,
                    labels_ignore=labels_ignore)

            pred = data_sample['pred_instances']
            result = dict(
                img_id=data_sample.get('img_id', len(self.results)),
                bboxes=_normalize_box_array(pred['bboxes']),
                scores=_to_numpy(pred['scores']),
                labels=_to_numpy(pred['labels']).astype(np.int64))

            self.results.append((ann, result))

    def fast_eval_recall(self,
                         gts: List[dict],
                         results: List[dict],
                         proposal_nums: Sequence[int],
                         iou_thrs: Sequence[float],
                         logger: Optional[MMLogger] = None) -> np.ndarray:
        """Evaluate proposal recall directly from DIOR-R samples.

        Args:
            gts (List[dict]): Ground truth annotations.
            results (List[dict]): Results of the dataset.
            proposal_nums (Sequence[int]): Proposal numbers used for
                evaluation.
            iou_thrs (Sequence[float]): IoU thresholds used for evaluation.
            logger (MMLogger, optional): Logger used for logging the recall
                summary.
        Returns:
            np.ndarray: Averaged recall results.
        """
        gt_bboxes = []
        pred_bboxes = []
        for ann, result in zip(gts, results):
            gt_bboxes.append(ann['bboxes'])
            bboxes = result['bboxes']
            scores = result['scores'].reshape(-1, 1)
            if bboxes.shape[0] == 0:
                pred_bboxes.append(_empty_boxes(bboxes.shape[1] + 1))
            else:
                pred_bboxes.append(
                    np.hstack([bboxes, scores]).astype(
                        np.float32, copy=False))

        recalls = eval_recalls(
            gt_bboxes, pred_bboxes, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        eval_results = OrderedDict()

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    gts, preds, self.proposal_nums, self.iou_thrs,
                    logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

        return eval_results