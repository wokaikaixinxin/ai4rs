# Preparing RSAR Dataset

<!-- [DATASET] -->

**[RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_RSAR_Restricted_State_Angle_Resolver_and_Rotated_SAR_Benchmark_CVPR_2025_paper.html)**  
**[Offical RSAR Link](https://github.com/zhasion/RSAR)**

## Results

|      Backbone      |        Model        |  mAP  | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: | :------------: | :---: | :----: | :-------: | :--: | :-----: | :---------------: |
| ResNet50<br> (800,800) |  Rotated-<br>RetinaNet  | 27.65 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/1rL7oAZQPpYuiiGow63uj5Ej4CJuzOv1d/view?usp=sharing) \| [log](https://drive.google.com/file/d/1yWA7Mlum_4b6KqDc7NX4yiGqqE8uDxFY/view?usp=sharing) |
| ResNet50<br> (800,800) |      R3Det       | 30.50 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/r3det/r3det-le90_r50_fpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/17oGLjtdOg6tlpqcA7Li1-BArBpAPeJ_9/view?usp=sharing) \| [log](https://drive.google.com/file/d/10cgIxVkq-KsrmUVyq0bDrQLxUWxp8_NO/view?usp=sharing) |
| ResNet50<br> (800,800) |      S2ANet      | 33.11 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/s2anet/s2anet-le90_r50_fpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/1xju1PGARP8h767Xr0yNpxNlan8E8hezJ/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Nr8QkDlrMmT7rJNlFIDfSDksZoAcRBX2/view?usp=sharing) |
| ResNet50<br> (800,800) |    Rotated-FCOS     | 34.22 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/rotated_fcos/rotated-fcos-le90_r50_fpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/13yswgvxNclZboOVy2x5pf7zdBWn5Q3yA/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Q53FL9WVWRxNuQ6_VqxvohHHVpZMmHxE/view?usp=sharing) |
| ResNet50<br> (800,800) | Rotated-Faster RCNN | 30.46 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/1ynmuD1Szq5KnOWlX86a-SBIe09yiXcbj/view?usp=sharing) \| [log](https://drive.google.com/file/d/1TxsS-pavIb8MDLxSRPpcGwV3WwSGxfeq/view?usp=sharing) |
| ResNet50<br> (800,800) |       O-RCNN        | 33.62 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/1xSUr6QOj8nyoQSmO2pmIqgvZEofwcQ7u/view?usp=sharing) \| [log](https://drive.google.com/file/d/1V3JroJK0B1_R1n2HguxBxYLvDkRKQCMV/view?usp=sharing) |
| ReResNet50<br> (800,800) |        ReDet        | 34.30 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/redet/redet-le90_re50_refpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/15z4WVeq4ChYoRXwvh_Nry4Ok9ozxytvB/view?usp=sharing) \| [log](https://drive.google.com/file/d/1P73YiWGWsPNSgu06kDqyB0cuW8nOe6oT/view?usp=sharing) |
| ResNet50<br> (800,800) |   RoI-Transformer   | 35.02 | `le90` |   `1x`    |  8=4gpu*<br>2img/gpu   | [config](configs/roi_trans/roi-trans-le90_r50_fpn_1x_rsar.py) | [ckpt](https://drive.google.com/file/d/1hmjnirDacJSqhTKolpKcnDeJxE5agU4U/view?usp=sharing) \| [log](https://drive.google.com/file/d/1NP-9wXuZVJymnpr_wmvRTQEyUpUPY9pM/view?usp=sharing) |
| ResNet50<br> (800,800) |   Deformable DETR   | 19.63 | `le90` | `3x` |  8=4gpu*<br>2img/gpu   | - | [ckpt](https://drive.google.com/file/d/10utd41U9R1UQAV1n4tJRkuNKRVDsjvPu/view?usp=sharing) \| [log](https://drive.google.com/file/d/1W5RWubZZNRwOEFcxw5bkfiq26Luxmf4R/view?usp=sharing) |
| ResNet50<br> (800,800) |      ARS-DETR       | 31.56 | `le90` | `3x` |  8=4gpu*<br>2img/gpu   | - | [ckpt](https://drive.google.com/file/d/1wKJZXn9Qb7RFN-B9pXQ_Jrx31_VGW4wd/view?usp=sharing) \| [log](https://drive.google.com/file/d/1NwJBLcnBCrRQupIbk9ixGtfhUPwb9EwL/view?usp=sharing) |

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