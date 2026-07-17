# Real-Time Oriented Object Detection Transformer in Remote Sensing Images (TGRS 2026)

[IEEE TGRS Xplore](https://ieeexplore.ieee.org/document/11424629)

[Arxiv](https://arxiv.org/abs/2603.15497)

[github Link](https://github.com/wokaikaixinxin/O2-RT-DETR)




Bilibili Install Tutorial:[![Bilibili](https://img.shields.io/badge/Installation_Tutorial-fb7299?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1Ufw4zyEhR/)
Train Tutorial: [![Bilibili](https://img.shields.io/badge/Train_Tutorial-fb7299?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1QQw4zJEot/)
Test Tutorial: [![Bilibili](https://img.shields.io/badge/Test_Tutorial-fb7299?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1Vew8zbEVa/)
Deploy Tutorial: [![Bilibili](https://img.shields.io/badge/Deploy_Tutorial-fb7299?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1VmwLzWExY/)

**NOTE: O2-RTDETR is earlier than YOLO 26 !!!**

## Abstract

Recent real-time detection transformers have gained popularity due to their simplicity and efficiency. However, these detectors do not explicitly model object rotation, especially in remote sensing imagery where objects appear at arbitrary angles, leading to challenges in angle representation, matching cost, and training stability. In this paper, **we propose a real-time oriented object detection transformer, the first real-time end-to-end oriented object detector to the best of our knowledge**, that addresses the above issues. Specifically, angle distribution refinement is proposed to reformulate angle regression as an iterative refinement of probability distributions, thereby capturing the uncertainty of object rotation and providing a more fine-grained angle representation. Then, we incorporate a Chamfer distance cost into bipartite matching, measuring box distance via vertex sets, enabling more accurate geometric alignment and eliminating ambiguous matches. Moreover, we propose oriented contrastive denoising to stabilize training and analyze four noise modes. We observe that a ground truth can be assigned to different index queries across different decoder layers, and analyze this issue using the proposed instability metric. We design a series of model variants and experiments to validate the proposed method.

## Main Results

**DOTA-v1.0 (Single-Scale Training and Testing)**

| Method | Backbone | AP50 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-RTDETR| R18vd (1024,1024,200) | 77.31 |    [o2_rtdetr_r18vd_2xb4_72e_dota](./configs/o2_rtdetr_r18vd_2xb4_72e_dota.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dota/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dota/20251011_200502/20251011_200502.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dota) |
|O2-RTDETR| R34vd (1024,1024,200) | 78.13 |    [o2_rtdetr_r34vd_2xb4_72e_dota](./configs/o2_rtdetr_r34vd_2xb4_72e_dota.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dota/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dota/20251011_195720/20251011_195720.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dota) |
|O2-RTDETR| R50vd (1024,1024,200) | 78.45 |    [o2_rtdetr_r50vd_2xb4_72e_dota](./configs/o2_rtdetr_r50vd_2xb4_72e_dota.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota/20251009_221010/20251009_221010.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dota) |

**DOTA-v1.0 (Multi-Scale Training and Testing)**

| Method | Backbone | AP50 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
| O2-DEIM | R18vd (1024,1024,200) | 79.49 | [o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms](./configs/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms.py) | [29th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms/epoch_29.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms/20251112_134456/20251112_134456.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms/29epoch.zip) |

```bash
# o2_rtdetr_r18vd_2xb4_72e_dota
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py your_ckpt 2

# o2_rtdetr_r34vd_2xb4_72e_dota
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dota.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dota.py your_ckpt 2

# o2_rtdetr_r50vd_2xb4_72e_dota
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py your_ckpt 2

# o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_coco_pretrain_72e_dota_ms.py your_ckpt 2
```


**DOTA-v1.5 (Single-Scale Training and Testing)**

| Method | Backbone | AP50 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-RTDETR| R18vd (1024,1024,200) | 70.83 |    [o2_rtdetr_r18vd_4xb1_72e_dotav15](./configs/o2_rtdetr_r18vd_4xb1_72e_dotav15.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_4xb1_72e_dotav15/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_4xb1_72e_dotav15/20251109_012504/20251109_012504.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_4xb1_72e_dotav15) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_4xb1_72e_dotav15/Task1.zip) |
|O2-RTDETR| R34vd (1024,1024,200) | 71.91 |    [o2_rtdetr_r34vd_4xb2_72e_dotav15](./configs/o2_rtdetr_r34vd_4xb2_72e_dotav15.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15/20251109_023554/20251109_023554.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_4xb2_72e_dotav15/Task1.zip) |
|O2-RTDETR| R50vd (1024,1024,200) | 73.76 |    [o2_rtdetr_r50vd_4xb2_72e_dotav15](./configs/o2_rtdetr_r50vd_4xb2_72e_dotav15.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15/20251108_230750/20251108_230750.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_4xb2_72e_dotav15/Task1.zip) |

```bash
# o2_rtdetr_r18vd_4xb1_72e_dotav15
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_4xb1_72e_dotav15.py 4
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_4xb1_72e_dotav15.py your_ckpt 4

# o2_rtdetr_r34vd_4xb2_72e_dotav15
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_4xb2_72e_dotav15.py 4
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_4xb2_72e_dotav15.py your_ckpt 4

# o2_rtdetr_r50vd_4xb2_72e_dotav15
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_4xb2_72e_dotav15.py 4
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_4xb2_72e_dotav15.py your_ckpt 4
```

**DIOR-R (Single-Scale Training and Testing)**

| Method | Backbone | AP50 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-RTDETR| R18vd (800,800) | 67.00 |    [o2_rtdetr_r18vd_2xb4_72e_dior](./configs/o2_rtdetr_r18vd_2xb4_72e_dior.py)      |  [72th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dior/epoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dior/20251012_211544.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb4_72e_dior) |
|O2-RTDETR| R34vd (800,800) | 68.67 |     [o2_rtdetr_r34vd_2xb4_72e_dior](./configs/o2_rtdetr_r34vd_2xb4_72e_dior.py)      | [48th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dior/epoch_48.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dior/20251012_211102/20251012_211102.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_dior)|
|O2-RTDETR| R50vd (800,800) | 72.26 |    [o2_rtdetr_r50vd_2xb4_72e_dior](./configs/o2_rtdetr_r50vd_2xb4_72e_dior.py)      |  [42th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dior/epoch_42.pth)\| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dior/20251013_100528/20251013_100528.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_dior) |

```bash
# o2_rtdetr_r18vd_2xb4_72e_dior
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dior.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dior.py your_cpkt 2

# o2_rtdetr_r34vd_2xb4_72e_dior
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dior.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dior.py your_cpkt 2

# o2_rtdetr_r50vd_2xb4_72e_dior
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dior.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dior.py your_cpkt 2
```

**FAIR1M-v1.0 (Single-Scale Training and Testing)**

| Method | Backbone | AP50 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|O2-RTDETR| R34vd (1024,1024,200) | 40.45 |    [o2_rtdetr_r34vd_2xb4_72e_fair1m](./configs/o2_rtdetr_r34vd_2xb4_72e_fair1m.py)      |  [24th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m/epoch_24.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m/20251110_215052/20251110_215052.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb4_72e_fair1m/fair1m_24_epoch.zip) |
|O2-RTDETR| R50vd (1024,1024,200) | 43.14 |    [o2_rtdetr_r50vd_2xb4_72e_fair1m](./configs/o2_rtdetr_r50vd_2xb4_72e_fair1m.py)      |  [18th_epoch](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m/epoch_18.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m/20251110_214150/20251110_214150.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m) \| [submit](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb4_72e_fair1m/fair1m_epoch_18.zip) |


```bash
# o2_rtdetr_r34vd_2xb4_72e_fair1m
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_fair1m.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_fair1m.py your_ckpt 2

# o2_rtdetr_r50vd_2xb4_72e_fair1m
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_fair1m.py 2
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_fair1m.py your_ckpt 2
```

Note: O2-RTDETR adopts single-scale training and testing on FAIR1M-v1.0, whereas several other methods (e.g., LSKNet, Strip R-CNN, and LegNet) report results based on multi-scale training and testing.

Note: We observed an interesting phenomenon: when training on FAIR1M-v1.0, the model exhibits a significant overfitting tendency on the training set.


**DroneVehicle**

| Method  | Modal | Backbone        | AP50  | mAP  | ep. |  bs  | Config  | Download |
| :-----: | :-----:  | :------:        | :--:  | :--: |  :--: | :--: | :-----: | :------: |
|O2-RTDETR|  RGB     | R18vd (640,512) | 70.95 | 44.86|  36   |  16  | [o2_rtdetr_r18vd_2xb8_36e_dv_rgb](./configs/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb/20260323_144051/20260323_144051.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb) |
|O2-RTDETR|  RGB     | R34vd (640,512) | 72.65 | 45.97 |  36   |  16  | [o2_rtdetr_r34vd_2xb8_36e_dv_rgb](./configs/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb/20260323_213225/20260323_213225.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_rgb) |
|O2-RTDETR|  RGB     | R50vd (640,512) | 73.69 | 47.37 |  36   |  16  | [o2_rtdetr_r50vd_2xb8_36e_dv_rgb](./configs/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb/20260323_213937/20260323_213937.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_rgb) |
|O2-RTDETR|  IR     | R18vd (640,512) | 72.73 | 48.29 |  36   |  16  | [o2_rtdetr_r18vd_2xb8_36e_dv_ir](./configs/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir/20260325_004850/20260325_004850.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_ir) |
|O2-RTDETR|  IR     | R34vd (640,512) | 72.82 | 48.48 |  36   |  16  | [o2_rtdetr_r34vd_2xb8_36e_dv_ir](./configs/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir/20260325_114240/20260325_114240.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r34vd_2xb8_36e_dronevehicle_ir) |
|O2-RTDETR|  IR     | R50vd (640,512) | 74.58 | 50.34 |  36   |  16  | [o2_rtdetr_r50vd_2xb8_36e_dv_ir](./configs/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir/epoch_36.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir/20260324_150848/20260324_150848.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r50vd_2xb8_36e_dronevehicle_ir) |



```bash
# for example
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb8_36e_dronevehicle_rgb.py 2
```

Note: These models (dronevehicle) are trained on the training set and evaluated on the test set, without using a validation set.


**RSAR**

| Backbone | Model | mAP | AP50 | AP75 | ep. | bs | Config | Download |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| Res18vd (800,800) | O2-RTDETR | 39.54 | 73.31 | 36.92 | 36 | 16 | [o2_rtdetr_r18vd_2xb8_36e_rsar](./configs/o2_rtdetr_r18vd_2xb8_36e_rsar.py) | [36th](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_rsar/epoch_36.pth) \| [all ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_rsar) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/o2_rtdetr/o2_rtdetr_r18vd_2xb8_36e_rsar/20260325_205614/20260325_205614.log) |

```bash
# for example
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb8_36e_rsar.py 2
```

## Visualization Results Demo

```bash
python demo/image_demo.py \
    demo/demo.jpg \
    projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py \
    your_path/epoch_72.pth \
    --out-file /root/demo_result.jpg
```

## Params, FLOPs and MACs
```bash
PYTHONPATH=. python tools/analysis_tools/get_flops/get_flops.py \
    --config projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py \
    --shape 1024 1024

PYTHONPATH=. python tools/analysis_tools/get_flops/get_flops.py \
    --config projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dota.py \
    --shape 1024 1024

PYTHONPATH=. python tools/analysis_tools/get_flops/get_flops.py \
    --config projects/rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_dota.py \
    --shape 1024 1024
```

## Deploy

[deploy_o2_rtdetr.md](../easydeploy/deploy_o2_rtdetr.md)

Bilibili Deploy Tutorial: [![Bilibili](https://img.shields.io/badge/Deploy_Tutorial-fb7299?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1VmwLzWExY/)

# Bibtex

```bibtex
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


<div align="center">
  <img src="https://github.com/wokaikaixinxin/O2-RT-DETR/blob/main/latency.jpg"  width="500"/>
</div>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=wokaikaixinxin/ai4rs/blob/main/projects/rotated_rtdetr/README.md)