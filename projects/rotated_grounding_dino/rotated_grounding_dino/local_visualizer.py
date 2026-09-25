from collections.abc import Sequence
from typing import List, Optional
import textwrap

import mmcv
import numpy as np
import torch
from mmdet.visualization import DetLocalVisualizer, jitter_color
from mmdet.visualization.palette import _get_adaptive_scales
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from torch import Tensor

from mmrotate.visualization import get_palette
from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes


class VGLocalVisualizer(DetLocalVisualizer):

    def _draw_instances(self, image: np.ndarray, data_sample: DetDataSample,
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]],
                        instances: Optional[InstanceData] = None) -> np.ndarray:
        """Draw visual grounding instances.

        Args:
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`): Data sample with text and instances.
            classes (List[str], optional): Unused for visual grounding.
            palette (List[tuple], optional): Palette for bbox color.

        Returns:
            np.ndarray: The drawn image which channel is RGB.
        """
        self.set_image(image)

        if instances is None:
            instances = data_sample.gt_instances

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            text = data_sample.text

            if isinstance(text, (list, tuple)):
                text = '. '.join(str(item) for item in text)
            else:
                text = str(text)

            bbox_color = self.bbox_color
            if bbox_color is None:
                bbox_color = palette[0] if palette is not None else (220, 20, 60)

            text_color = get_palette(self.text_color, 1)[0]
            color = get_palette(bbox_color, 1)[0]

            if isinstance(bboxes, Tensor):
                if bboxes.size(-1) == 5:
                    bboxes = RotatedBoxes(bboxes)
                elif bboxes.size(-1) == 8:
                    bboxes = QuadriBoxes(bboxes)
                else:
                    raise TypeError(
                        'Require the shape of `bboxes` to be (n, 5) '
                        'or (n, 8), but get `bboxes` with shape being '
                        f'{bboxes.shape}.')

            bboxes = bboxes.cpu()
            polygons = bboxes.convert_to('qbox').tensor
            polygons = polygons.reshape(-1, 4, 2)
            polygons = [p for p in polygons]

            self.draw_polygons(
                polygons,
                edge_colors=[color for _ in polygons],
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes.centers + self.line_width
            # scales = _get_adaptive_scales(bboxes.areas)

            for i, pos in enumerate(positions):
                label_text = text
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_color,
                    font_sizes=int(8),
                    bboxes=[{
                        'facecolor': 'gray',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample,
                                                   classes, palette)
            if 'gt_sem_seg' in data_sample:
                raise NotImplementedError

            if 'gt_panoptic_seg' in data_sample:
                raise NotImplementedError

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                if 'scores' in pred_instances and len(pred_instances) > 0:
                    top1_idx = int(pred_instances.scores.argmax())
                    pred_instances = pred_instances[top1_idx:top1_idx + 1]
                pred_img_data = self._draw_instances(image, data_sample,
                                                     classes, palette,
                                                     pred_instances)

            if 'pred_sem_seg' in data_sample:
                raise NotImplementedError

            if 'pred_panoptic_seg' in data_sample:
                raise NotImplementedError

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)