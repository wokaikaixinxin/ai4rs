# ReDiffDet: Rotation-equivariant Diffusion Model for Oriented Object Detection

> [ReDiffDet: Rotation-equivariant Diffusion Model for Oriented Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_ReDiffDet_Rotation-equivariant_Diffusion_Model_for_Oriented_Object_Detection_CVPR_2025_paper.html)

<!-- [ALGORITHM] -->

## Insight

- 模型能将随机输入学习得到目标。随机输入都可行，其他类型的输入大概率也是可行的。
- 旋转框可以表征为概率分布，如高斯分布。万物都可以表征为不同的概率分布。
- 扩散模型、高斯泼溅等都涉及概率分布，都涉及随机性。
- 架构是decoder-only型，由多层decoder layer堆叠。代码借用了openmmlab 中two-stage类作为父类，事实上，将one-stage或transformer作为父类都可以。
- 这是ReDiffDet的baseline。

- The model can learn targets from random inputs. If random inputs work, other types of inputs are likely feasible as well.  
- Oriented boxes can be represented as probability distributions, such as Gaussian distributions. Everything can be characterized by different probability distributions.  
- Diffusion models, Gaussian splatting, etc., all involve probability distributions and randomness.  
- The architecture is decoder-only, stacked with multiple decoder layers. The code borrows the two-stage class from OpenMMLab as the parent class, though using one-stage or Transformer as the parent is also possible.
- This is the baseline of ReDiffDet.

## New

&#x2705; 旋转卡壳算法-求点集的最小外接矩形  
&#x2705; Rotating Calipers Algorithm - Finding the Minimum Bounding Rectangle of a Point Set

## Abstract

<div align=center>
<img src="https://github.com/wokaikaixinxin/ReDiffDet/blob/main/rediffdet_overview.png" width="800"/>
</div>

The diffusion model has been successfully applied to various detection tasks. However, it still faces several challenges when used for oriented object detection: objects that are arbitrarily rotated require the diffusion model to encode their orientation information; uncontrollable random boxes inaccurately locate objects with dense arrangements and extreme aspect ratios; oriented boxes result in the misalignment between them and image features. To overcome these limitations, we propose ReDiffDet, a framework that formulates oriented object detection as a rotation-equivariant denoising diffusion process. First, we represent an oriented box as a 2D Gaussian distribution, forming the basis of the denoising paradigm. The reverse process can be proven to be rotation-equivariant within this representation and model framework. Second, we design a conditional encoder with conditional boxes to prevent boxes from being randomly placed across the entire image. Third, we propose an aligned decoder for alignment between oriented boxes and image features. The extensive experiments demonstrate ReDiffDet achieves promising performance and significantly outperforms the diffusion-based baseline detector.

## Results and models

**RSAR**
|      Backbone      |        Model        |  mAP  |  AP50 | AP75 | Angle  |  lr schd  |  BS  | Config | Download |
| :----------: | :------------: | :---: | :----: | :----: | :----: |:-------: | :--: | :-----: | :---------------: |
| *Random* |  |  |  | | |  | |  |
| ResNet50<br> (800,800) |  [GSDet](../../../projects/GSDet_baseline/README_GSDet_baseline.md)     | *35.58* | *68.00* | *33.50* | `le90` | `1x` |  4=2gpu*<br>2img/gpu   | [config](../../../projects/GSDet_baseline/configs/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar.py) | [best ckpt]() \| <br> [all ckpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| <br> [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar/20250712_230231.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_1x_rsar/20250714_164452.log) |

**NOTE: the mAP, AP50, and AP75 are reported on test set, not val set !!!**

**DOTA1.0**


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 47.69 | 73.88  |   52.17    |   le90   |      2x      |  -  | 4=2gpu*<br>2img/gpu      | [GSDet_r50_b900_h2h4<br>_h2r1_r2r1_2x_dotav1.0.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0/20250703_225830/20250703_225830.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0/Task1.zip) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7387758407976989  
ap of each class:   
plane:0.8850394908129608, baseball-diamond:0.7901977747706781, bridge:0.5057504797851325, ground-track-field:0.7170557083792797, small-vehicle:0.7852998720627247, large-vehicle:0.8320931918879327, ship:0.8824600057919189, tennis-court:0.9084433118119101, basketball-court:0.8425974293353717, storage-tank:0.8032986565311625, soccer-ball-field:0.5587179317542539, roundabout:0.5704119054786441, harbor:0.7469883448696094, swimming-pool:0.6787017814257459, helicopter:0.5745817272681583  
COCO style result:  
AP50: 0.7387758407976989  
AP75: 0.521672244119724  
mAP: 0.47687469541788785



**DIOR-R**


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800,200) | 46.34 | 69.80  |   48.16    |   le90   |      2x      |  -  | 4=2gpu*<br>2img/gpu      | [GSDet_r50_b900_h2h4<br>_h2r1_r2r1_2x_dior.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior/20241208_124846/20241208_124846.log) \| [results](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior/20250704_182754/20250704_182754.log) |


---------------iou_thr: 0.5---------------

| class                   | gts   | dets   | recall  | ap      |
|-------------------------|-------|--------|---------|---------|
| airplane                | 8212  | 146038 | 0.88894 | 0.78397 |
| airport                 | 666   | 93561  | 0.93393 | 0.60829 |
| baseballfield           | 3434  | 115193 | 0.88701 | 0.76466 |
| basketballcourt         | 2146  | 257083 | 0.94688 | 0.87331 |
| bridge                  | 2589  | 507543 | 0.76477 | 0.48336 |
| chimney                 | 1031  | 128298 | 0.87294 | 0.77658 |
| expressway-service-area | 1085  | 86583  | 0.95668 | 0.86513 |
| expressway-toll-station | 688   | 133752 | 0.86192 | 0.73194 |
| dam                     | 538   | 122793 | 0.93866 | 0.45327 |
| golffield               | 575   | 118450 | 0.94435 | 0.77543 |
| groundtrackfield        | 1885  | 115998 | 0.97082 | 0.79884 |
| harbor                  | 3105  | 211913 | 0.69147 | 0.40581 |
| overpass                | 1782  | 210588 | 0.81481 | 0.60808 |
| ship                    | 35186 | 179220 | 0.90243 | 0.83317 |
| stadium                 | 672   | 58144  | 0.95982 | 0.76221 |
| storagetank             | 23361 | 235388 | 0.80382 | 0.72764 |
| tenniscourt             | 7343  | 139576 | 0.91924 | 0.86866 |
| trainstation            | 509   | 74991  | 0.91945 | 0.57726 |
| vehicle                 | 26640 | 872233 | 0.70653 | 0.55803 |
| windmill                | 2998  | 118197 | 0.88559 | 0.70465 |
| mAP                     |       |        |         | 0.69801 |


DIOR-R/mAP: 0.4634  DIOR-R/AP50: 0.6980  DIOR-R/AP55: 0.6686  DIOR-R/AP60: 0.6336  DIOR-R/AP65: 0.5934  DIOR-R/AP70: 0.5386  DIOR-R/AP75: 0.4816  DIOR-R/AP80: 0.4088  DIOR-R/AP85: 0.3207  DIOR-R/AP90: 0.2123  DIOR-R/AP95: 0.0781

## Citation

```
@inproceedings{zhao2025rediffdet,
  title={ReDiffDet: Rotation-equivariant Diffusion Model for Oriented Object Detection},
  author={Zhao, Jiaqi and Ding, Zeyu and Zhou, Yong and Zhu, Hancheng and Du, Wen-Liang and Yao, Rui},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={24429--24439},
  year={2025}
}
```
