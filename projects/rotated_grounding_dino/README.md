# A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing -- O2-VG-Trans



[Arxiv](https://arxiv.org/abs/2609.28230)





## Abstract

Visual grounding in remote sensing images aims to locate objects described by referring expressions. Most existing methods predict horizontal bounding boxes, which are often inaccurate for objects with arbitrary orientations. To address this limitation, we introduce O2-VG, a family of models for oriented object visual grounding with three complementary designs. Specifically, O2-VG-Trans is a cross-modality transformer for oriented object visual grounding. It establishes a strong discriminative foundation for the model family. Building upon it, O2-VG-Uni predicts universal oriented proposals for possible foreground objects without specific text prompts. It also supports object retrieval through cached proposal embeddings. Using these universal oriented proposals as input prompts, O2-VG-VLM is an autoregressive vision-language model. It generates oriented box token blocks in parallel through multi-token prediction. In addition, we construct DIOR-R-RSVG, a dataset for oriented object visual grounding in remote sensing images. It provides image, expression, and oriented box triplets for training and evaluation. Together, the O2-VG family provides a flexible framework that spans discriminative transformers and generative vision-language models. It achieves superior performance across multiple benchmarks.


<div align=center>
<img src='https://github.com/wokaikaixinxin/Eagle_o2_vg/blob/main/asset/overview.png' width="80%"/>
</div>

<div align=center>
<img src='https://github.com/wokaikaixinxin/Eagle_o2_vg/blob/main/asset/o2_vg_trans.png' width="80%"/>
</div>


## Preparation -- BERT Model

Down load BERT

```
# you can set hugging face mirror
export HF_ENDPOINT=https://hf-mirror.com
# download
huggingface-cli download google-bert/bert-base-uncased --local-dir /root/bert-base-uncased
```
Modified BERT Path in [avvg.py](./configs/avvg.py#L3), [dior_r_rsvg.py](./configs/dior_r_rsvg.py#L3), and [vrsbench.py](./configs/vrsbench.py#L3)

## Installation

```shell
pip install fairscale -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Main Results

**DIOR-R-RSVG**

| Method | Backbone | Pr@0.5 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-VG-Trans| R50 (800,800) | 67.23 |    [grounding_dino_r50_bs8_1x_dior_r_rsvg](./configs/grounding_dino_r50_bs8_1x_dior_r_rsvg.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_grounding_dino%2Fgrounding_dino_r50_bs8_1x_dior_r_rsvg%2Fepoch_12.pth) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_grounding_dino/grounding_dino_r50_bs8_1x_dior_r_rsvg) |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|O2-VG-Trans| R50 (800,800) | 67.23 | 62.00 | 54.12 | 39.47 | 17.01 | 56.73 | 67.33 |

train

```Shell
bash tools/dist_train.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs8_1x_dior_r_rsvg.py 2
```

test

```Shell
bash tools/dist_test.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs8_1x_dior_r_rsvg.py work_dirs/grounding_dino_r50_bs8_1x_dior_r_rsvg/epoch_12.pth 2
```

**VRSBench**

| Method | Backbone | Pr@0.5 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-VG-Trans| R50 (512,512) | 67.71 |    [grounding_dino_r50_bs8_1x_vrsbench](./configs/grounding_dino_r50_bs8_1x_vrsbench.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_grounding_dino%2Fgrounding_dino_r50_bs8_1x_vrsbench%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_grounding_dino/grounding_dino_r50_bs8_1x_vrsbench/20260807_000901/20260807_000901.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_grounding_dino/grounding_dino_r50_bs8_1x_vrsbench) |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|O2-VG-Trans| R50 (512,512) | 67.71 | 60.13 | 46.53 | 27.11 | 7.63 | 55.01 | 61.14 |

train

```Shell
bash tools/dist_train.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs8_1x_vrsbench.py 2
```
test

```Shell
bash tools/dist_test.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs8_1x_vrsbench.py work_dirs/grounding_dino_r50_bs8_1x_vrsbench/epoch_12.pth 2
```

**AVVG**

| Method | Backbone | Pr@0.5 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-VG-Trans| R50 (1024,576) | 18.00 |    [grounding_dino_r50_bs2_1x_avvg](./configs/grounding_dino_r50_bs2_1x_avvg.py)      |  [11th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_grounding_dino%2Fgrounding_dino_r50_bs2_1x_avvg%2Fepoch_11.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_grounding_dino/grounding_dino_r50_bs2_1x_avvg/20260807_155302/20260807_155302.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_grounding_dino/grounding_dino_r50_bs2_1x_avvg) |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|O2-VG-Trans| R50 (1024,576) | 18.00 | 17.04 | 14.37 | 7.83 | 0.98 | 14.59 | 16.64 |

train

```Shell
bash tools/dist_train.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs2_1x_avvg.py 2
```

test

```Shell
bash tools/dist_test.sh projects/rotated_grounding_dino/configs/grounding_dino_r50_bs2_1x_avvg.py work_dirs/grounding_dino_r50_bs2_1x_avvg/epoch_11.pth 2
```

## Citation

```bibtex
@article{ding2026unified,
  title={A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing},
  author={Ding, Zeyu and Zhou, Yong and Zhao, Jiaqi and Du, Wen-Liang and
          Li, Xixi and Zhu, Hancheng and Yao, Rui and El Saddik, Abdulmotaleb},
  journal={arXiv preprint arXiv:2609.28230},
  year={2026}
}
``` 

## Acknowledgements

The **first** real-time oriented object detection transformer: 

Real-Time Oriented Object Detection Transformer in Remote Sensing Images

[O2-RTDETR github](../rotated_rtdetr/README.md)

```
@ARTICLE{11424629,
  author={Ding, Zeyu and Zhou, Yong and Zhao, Jiaqi and Du, Wen-Liang and Li, Xixi and Yao, Rui and Saddik, Abdulmotaleb El},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Real-Time Oriented Object Detection Transformer in Remote Sensing Images}, 
  year={2026},
  volume={64},
  number={5613014},
  pages={1-14},
  keywords={Real-time systems;Transformers;Detectors;Remote sensing;Costs;Training;Accuracy;YOLO;Uncertainty;Noise reduction;Detection transformer (DETR);oriented object detection;real-time detector;remote sensing},
  doi={10.1109/TGRS.2026.3671683}}
```

