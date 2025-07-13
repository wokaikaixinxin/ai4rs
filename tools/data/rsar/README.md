# Preparing RSAR Dataset

<!-- [DATASET] -->

**RSAR paper link :**  
[RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_RSAR_Restricted_State_Angle_Resolver_and_Rotated_SAR_Benchmark_CVPR_2025_paper.html)

**RSAR official code link :**   
[Offical RSAR Link](https://github.com/zhasion/RSAR)

## Results
**Table 4 in Paper**

|      Backbone      |        Model        |  mAP  |  AP50 | AP75 | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: | :------------: | :---: | :----: | :----: | :----: |:-------: | :--: | :-----: | :---------------: |
| One-Stage |  |  |  | | |  | |  |
| ResNet50<br> (800,800) |  Rotated-<br>RetinaNet  | 27.65 | 57.67 |  22.72  | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar.json) |
| ResNet50<br> (800,800) |      R3Det       | 30.50 | 63.94 | 25.02 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/r3det/r3det-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/r3det-le90_r50_fpn_1x_rsar/r3det-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/s2anet-le90_r50_fpn_1x_rsar/s2anet-le90_r50_fpn_1x_rsar.json) |
| ResNet50<br> (800,800) |      S2ANet      | 33.11 | 66.47 | 28.52 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/s2anet/s2anet-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/s2anet-le90_r50_fpn_1x_rsar/s2anet-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://drive.google.com/file/d/1Nr8QkDlrMmT7rJNlFIDfSDksZoAcRBX2/view?usp=sharing) |
| ResNet50<br> (800,800) |    Rotated-<br>FCOS     | 34.22 | 66.66 | 31.45 |`le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/rotated_fcos/rotated-fcos-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-fcos-le90_r50_fpn_1x_rsar/rotated-fcos-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-fcos-le90_r50_fpn_1x_rsar/rotated-fcos-le90_r50_fpn_1x_rsar.json) |
| Two-Stage |  |  |  | | |  | |  |
| ResNet50<br> (800,800) | Rotated-Faster <br> RCNN | 30.46 | 63.18 | 24.88 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-faster-rcnn-le90_r50_fpn_1x_rsar/rotated-faster-rcnn-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/rotated-faster-rcnn-le90_r50_fpn_1x_rsar/rotated-faster-rcnn-le90_r50_fpn_1x_rsar.json) |
| ResNet50<br> (800,800) |       O-RCNN        | 33.62 | 64.82 | 32.69 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/oriented-rcnn-le90_r50_fpn_1x_rsar/oriented-rcnn-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/oriented-rcnn-le90_r50_fpn_1x_rsar/oriented-rcnn-le90_r50_fpn_1x_rsar_epoch_12.pth) |
| ReResNet50<br> (800,800) |        ReDet        | 34.30 | 64.71 | 32.84 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/redet/redet-le90_re50_refpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/redet-le90_re50_refpn_1x_rsar/redet-le90_re50_refpn_1x_rsar_epoch_12_.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/redet-le90_re50_refpn_1x_rsar/redet-le90_re50_refpn_1x_rsar.json) |
| ResNet50<br> (800,800) |   RoI-Transformer   | 35.02 | 66.95 | 32.65 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](../../../configs/roi_trans/roi-trans-le90_r50_fpn_1x_rsar.py) | [ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/roi-trans-le90_r50_fpn_1x_rsar/roi-trans-le90_r50_fpn_1x_rsar_epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/RSAR/roi-trans-le90_r50_fpn_1x_rsar/roi-trans-le90_r50_fpn_1x_rsar.json) |
| Query |  |  |  | | |  | |  |
| ResNet50<br> (800,800) |   Deformable DETR   | 19.63 | 46.62 | 13.06 | `le90` | `3x` |  8=4gpu*<br>2img/gpu   | - | - |
| ResNet50<br> (800,800) |      ARS-DETR       | 31.56 | 61.14 | 28.97 | `le90` | `3x` |  8=4gpu*<br>2img/gpu   | - | - |

## Download RSAR dataset

The RSAR dataset can be downloaded from [ModelScope (魔塔)](https://www.modelscope.cn/datasets/wokaikaixinxin/RSAR/files)

The data structure is as follows:

```none
ai4rs
├── data
│   ├── RSAR
│   │   ├── train
│   │   │   ├── annfiles
│   │   │   ├── images
│   │   ├── val
│   │   │   ├── annfiles
│   │   │   ├── images
│   │   ├── test
│   │   │   ├── annfiles
│   │   │   ├── images
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/rsar.py` to `data/rsar/`.


## Cite
```bibtex
@inproceedings{zhang2025rsar,
  title={Rsar: Restricted state angle resolver and rotated sar benchmark},
  author={Zhang, Xin and Yang, Xue and Li, Yuxuan and Yang, Jian and Cheng, Ming-Ming and Li, Xiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={7416--7426},
  year={2025}
}
```