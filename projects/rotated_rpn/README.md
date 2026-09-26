# A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing proposes the Rotated RPN


[Arxiv](https://arxiv.org/abs/2609.28230)



## Abstract

Visual grounding in remote sensing images aims to locate objects described by referring expressions. Most existing methods predict horizontal bounding boxes, which are often inaccurate for objects with arbitrary orientations. To address this limitation, we introduce O2-VG, a family of models for oriented object visual grounding with three complementary designs. Specifically, O2-VG-Trans is a cross-modality transformer for oriented object visual grounding. It establishes a strong discriminative foundation for the model family. Building upon it, O2-VG-Uni predicts universal oriented proposals for possible foreground objects without specific text prompts. It also supports object retrieval through cached proposal embeddings. Using these universal oriented proposals as input prompts, O2-VG-VLM is an autoregressive vision-language model. It generates oriented box token blocks in parallel through multi-token prediction. In addition, we construct DIOR-R-RSVG, a dataset for oriented object visual grounding in remote sensing images. It provides image, expression, and oriented box triplets for training and evaluation. Together, the O2-VG family provides a flexible framework that spans discriminative transformers and generative vision-language models. It achieves superior performance across multiple benchmarks.


## Main Results

**DIOR-R-RSVG**

| Method | Backbone |  Config | Download |
| :----: | :------: |  :-----: | :------: |
|rotated rpn| R18 (800,800) |    [oriented_rpn_r18_fpn_1x_dior_r_rsvg.py](./configs/oriented_rpn_r18_fpn_1x_dior_r_rsvg.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r18_fpn_1x_dior_r_rsvg%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r18_fpn_1x_dior_r_rsvg%2F20260830_204638%2F20260830_204638.log) \|[all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_rpn/oriented_rpn_r18_fpn_1x_dior_r_rsvg) |
|rotated rpn| R50 (800,800) |  [oriented_rpn_r50_fpn_1x_dior_r_rsvg.py](./configs/oriented_rpn_r50_fpn_1x_dior_r_rsvg.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r50_fpn_1x_dior_r_rsvg%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r50_fpn_1x_dior_r_rsvg%2F20260830_165022%2F20260830_165022.log) \|[all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_rpn/oriented_rpn_r50_fpn_1x_dior_r_rsvg) |


rotated rpn resnet18 on DIOR-R-RSVG:

|      | 0.5     | 0.55    | 0.6     | 0.65    | 0.7     | 0.75    | 0.8     | 0.85    | 0.9     | 0.95    |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 100  | 0.85080 | 0.81733 | 0.77187 | 0.70600 | 0.61293 | 0.50360 | 0.36067 | 0.19573 | 0.06427 | 0.00680 |
| 300  | 0.89800 | 0.87000 | 0.82453 | 0.76053 | 0.66253 | 0.54693 | 0.39027 | 0.21133 | 0.06693 | 0.00707 |

AR50:95@100	0.48900
AR50:95@300	0.52381

rotated rpn resnet50 on DIOR-R-RSVG:

|      | 0.5     | 0.55    | 0.6     | 0.65    | 0.7     | 0.75    | 0.8     | 0.85    | 0.9     | 0.95    |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 100  | 0.88213 | 0.85267 | 0.81000 | 0.74987 | 0.66333 | 0.54653 | 0.39933 | 0.22747 | 0.07733 | 0.00893 |
| 300  | 0.91720 | 0.89160 | 0.85213 | 0.79493 | 0.70853 | 0.58307 | 0.42547 | 0.23760 | 0.07947 | 0.00893 |

AR@100	0.52176
AR@300	0.54989


train

```Shell
bash tools/dist_train.sh projects/rotated_rpn/configs/oriented_rpn_r18_fpn_1x_dior_r_rsvg.py 2
bash tools/dist_train.sh projects/rotated_rpn/configs/oriented_rpn_r50_fpn_1x_dior_r_rsvg.py 2
```

test

```Shell
bash tools/dist_test.sh projects/rotated_rpn/configs/oriented_rpn_r18_fpn_1x_dior_r_rsvg.py work_dirs/oriented_rpn_r18_fpn_1x_dior_r_rsvg/epoch_12.pth 2
bash tools/dist_test.sh projects/rotated_rpn/configs/oriented_rpn_r50_fpn_1x_dior_r_rsvg.py work_dirs/oriented_rpn_r50_fpn_1x_dior_r_rsvg/epoch_12.pth 2
```

**VRSBench**

| Method | Backbone |   Config | Download |
| :----: | :------: |  :-----: | :------: |
|rotated rpn| R18 (512, 512) |    [oriented_rpn_r18_fpn_1x_vrsbench](./configs/oriented_rpn_r18_fpn_1x_vrsbench.py) |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r18_fpn_1x_vrsbench%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r18_fpn_1x_vrsbench%2F20260830_200649%2F20260830_200649.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_rpn/oriented_rpn_r18_fpn_1x_vrsbench) |
|rotated rpn| R50 (512, 512) |    [oriented_rpn_r50_fpn_1x_vrsbench](./configs/oriented_rpn_r50_fpn_1x_vrsbench.py)  |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r50_fpn_1x_vrsbench%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r50_fpn_1x_vrsbench%2F20260830_165025%2F20260830_165025.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_rpn/oriented_rpn_r50_fpn_1x_vrsbench) |

rotated rpn resnet18 on vrsbench:

| proposal | 0.5     | 0.55    | 0.6     | 0.65    | 0.7     | 0.75    | 0.8     | 0.85    | 0.9     | 0.95    |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 100  | 0.83136 | 0.79145 | 0.72956 | 0.64688 | 0.53555 | 0.40776 | 0.26648 | 0.13274 | 0.03633 | 0.00241 |
| 300  | 0.87772 | 0.84287 | 0.78854 | 0.70790 | 0.59094 | 0.45356 | 0.30095 | 0.14667 | 0.03837 | 0.00254 |

AR50:95@100	0.43805
AR50:95@300	0.47500

rotated rpn resnet50 on vrsbench:

|proposal| 0.5     | 0.55    | 0.6     | 0.65    | 0.7     | 0.75    | 0.8     | 0.85    | 0.9     | 0.95    |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 100  | 0.85791 | 0.82722 | 0.78198 | 0.71366 | 0.61860 | 0.49025 | 0.34538 | 0.18634 | 0.05483 | 0.00439 |
| 300  | 0.89777 | 0.86992 | 0.82777 | 0.76434 | 0.66898 | 0.53716 | 0.37552 | 0.20261 | 0.05898 | 0.00439 |

AR50:95@100	0.48806
AR50:95@300	0.52074

train

```Shell
bash tools/dist_train.sh projects/rotated_rpn/configs/oriented_rpn_r18_fpn_1x_vrsbench.py 2
bash tools/dist_train.sh projects/rotated_rpn/configs/oriented_rpn_r50_fpn_1x_vrsbench.py 2
```
test

```Shell
bash tools/dist_test.sh projects/rotated_rpn/configs/oriented_rpn_r18_fpn_1x_vrsbench.py work_dirs/oriented_rpn_r18_fpn_1x_vrsbench/epoch_12.pth 2
bash tools/dist_test.sh projects/rotated_rpn/configs/oriented_rpn_r50_fpn_1x_vrsbench.py work_dirs/oriented_rpn_r50_fpn_1x_vrsbench/epoch_12.pth 2
```

**AVVG**

| Method | Backbone |   Config | Download |
| :----: | :------: |  :-----: | :------: |
|rotated rpn| R18 (1024, 576) |    [oriented_rpn_r18_fpn_1x_avvg](./configs/oriented_rpn_r18_fpn_1x_avvg.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r18_fpn_1x_avvg%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r18_fpn_1x_avvg%2F20260830_193245%2F20260830_193245.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_rpn/oriented_rpn_r18_fpn_1x_avvg) |
|rotated rpn| R50 (1024, 576) |    [oriented_rpn_r50_fpn_1x_avvg](./configs/oriented_rpn_r50_fpn_1x_avvg.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r50_fpn_1x_avvg%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_rpn%2Foriented_rpn_r50_fpn_1x_avvg%2F20260830_170021%2F20260830_170021.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_rpn/oriented_rpn_r50_fpn_1x_avvg) |


rotated rpn resnet18 on avvg:

|proposal | 0.5     | 0.55    | 0.6     | 0.65    | 0.7     | 0.75    | 0.8     | 0.85    | 0.9     | 0.95    |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 100  | 0.54979 | 0.50503 | 0.44395 | 0.36166 | 0.25705 | 0.14864 | 0.06696 | 0.02011 | 0.00258 | 0.00012 |
| 300  | 0.80648 | 0.75791 | 0.68702 | 0.57677 | 0.42421 | 0.23976 | 0.10289 | 0.02919 | 0.00429 | 0.00025 |

AR50:95@100	0.23559
AR50:95@300	0.36288

rotated rpn resnet50 on avvg:

|proposal | 0.5     | 0.55    | 0.6     | 0.65    | 0.7     | 0.75    | 0.8     | 0.85    | 0.9     | 0.95    |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 100  | 0.604 | 0.564 | 0.500 | 0.420 | 0.302 | 0.191 | 0.087 | 0.027 | 0.005 | 0.000 |
| 300  | 0.865 | 0.826 | 0.756 | 0.650 | 0.482 | 0.300 | 0.138 | 0.042 | 0.008 | 0.000 |

AR50:95@100	0.2700
AR50:95@300	0.4067

train

```Shell
bash tools/dist_train.sh projects/rotated_rpn/configs/oriented_rpn_r18_fpn_1x_avvg.py 2
bash tools/dist_train.sh projects/rotated_rpn/configs/oriented_rpn_r50_fpn_1x_avvg.py 2
```

test

```Shell
bash tools/dist_test.sh projects/rotated_rpn/configs/oriented_rpn_r18_fpn_1x_avvg.py work_dirs/oriented_rpn_r18_fpn_1x_avvg/epoch_12.pth 2
bash tools/dist_test.sh projects/rotated_rpn/configs/oriented_rpn_r50_fpn_1x_avvg.py /root/mmrotate-1.x/work_dirs/oriented_rpn_r50_fpn_1x_avvg/epoch_12.pth 2
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

#1 Faster RCNN

```
@article{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  journal={Advances in neural information processing systems},
  volume={28},
  year={2015}
}
```

#2

Oriented RCNN

```
@inproceedings{xie2021oriented,
  title={Oriented R-CNN for object detection},
  author={Xie, Xingxing and Cheng, Gong and Wang, Jiabao and Yao, Xiwen and Han, Junwei},
  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={3500--3509},
  year={2021},
  organization={IEEE}
}
```