# A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing



[Arxiv](https://arxiv.org/abs/2609.28230)





## Abstract

Visual grounding in remote sensing images aims to locate objects described by referring expressions. Most existing methods predict horizontal bounding boxes, which are often inaccurate for objects with arbitrary orientations. To address this limitation, we introduce O2-VG, a family of models for oriented object visual grounding with three complementary designs. Specifically, O2-VG-Trans is a cross-modality transformer for oriented object visual grounding. It establishes a strong discriminative foundation for the model family. Building upon it, O2-VG-Uni predicts universal oriented proposals for possible foreground objects without specific text prompts. It also supports object retrieval through cached proposal embeddings. Using these universal oriented proposals as input prompts, O2-VG-VLM is an autoregressive vision-language model. It generates oriented box token blocks in parallel through multi-token prediction. In addition, we construct DIOR-R-RSVG, a dataset for oriented object visual grounding in remote sensing images. It provides image, expression, and oriented box triplets for training and evaluation. Together, the O2-VG family provides a flexible framework that spans discriminative transformers and generative vision-language models. It achieves superior performance across multiple benchmarks.

## Main Results

**DIOR-R-RSVG**

| Method | Backbone | Pr@0.5 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-VG-Trans| R50 (800,800) | 67.23 |    [grounding_dino_r50_bs8_1x_dior_r_rsvg](./configs/grounding_dino_r50_bs8_1x_dior_r_rsvg.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_grounding_dino%2Fgrounding_dino_r50_bs8_1x_dior_r_rsvg%2Fepoch_12.pth) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_grounding_dino/grounding_dino_r50_bs8_1x_dior_r_rsvg) |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|O2-VG-Trans| R50 (800,800) | 67.23 | 62.00 | 54.12 | 39.47 | 17.01 | 56.73 | 67.33 |

```Shell
bash tools/dist_train.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs8_1x_dior_r_rsvg.py 2
```

**VRSBench**

| Method | Backbone | Pr@0.5 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-VG-Trans| R50 (512,512) | 67.71 |    [grounding_dino_r50_bs8_1x_vrsbench](./configs/grounding_dino_r50_bs8_1x_vrsbench.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_grounding_dino%2Fgrounding_dino_r50_bs8_1x_vrsbench%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_grounding_dino/grounding_dino_r50_bs8_1x_vrsbench/20260807_000901/20260807_000901.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_grounding_dino/grounding_dino_r50_bs8_1x_vrsbench) |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|O2-VG-Trans| R50 (512,512) | 67.71 | 60.13 | 46.53 | 27.11 | 7.63 | 55.01 | 61.14 |


```Shell
bash tools/dist_train.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs8_1x_vrsbench.py 2
```


**AVVG**

| Method | Backbone | Pr@0.5 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-VG-Trans| R50 (1024,576) | 18.00 |    [grounding_dino_r50_bs2_1x_avvg](./configs/grounding_dino_r50_bs2_1x_avvg.py)      |  [11th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_grounding_dino%2Fgrounding_dino_r50_bs2_1x_avvg%2Fepoch_11.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_grounding_dino/grounding_dino_r50_bs2_1x_avvg/20260807_155302/20260807_155302.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_grounding_dino/grounding_dino_r50_bs2_1x_avvg) |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|O2-VG-Trans| R50 (1024,576) | 18.00 | 17.04 | 14.37 | 7.83 | 0.98 | 14.59 | 16.64 |


```Shell
bash tools/dist_train.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs2_1x_avvg.py 2
```


## Visualization Results Demo




## Bibtex

