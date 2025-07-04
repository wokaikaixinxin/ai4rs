# ReDiffDet: Rotation-equivariant Diffusion Model for Oriented Object Detection

> [ReDiffDet: Rotation-equivariant Diffusion Model for Oriented Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_ReDiffDet_Rotation-equivariant_Diffusion_Model_for_Oriented_Object_Detection_CVPR_2025_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/jhlmarques/GauCho/blob/main/images/concept_gaucho.png" width="800"/>
</div>

The diffusion model has been successfully applied to various detection tasks. However, it still faces several challenges when used for oriented object detection: objects that are arbitrarily rotated require the diffusion model to encode their orientation information; uncontrollable random boxes inaccurately locate objects with dense arrangements and extreme aspect ratios; oriented boxes result in the misalignment between them and image features. To overcome these limitations, we propose ReDiffDet, a framework that formulates oriented object detection as a rotation-equivariant denoising diffusion process. First, we represent an oriented box as a 2D Gaussian distribution, forming the basis of the denoising paradigm. The reverse process can be proven to be rotation-equivariant within this representation and model framework. Second, we design a conditional encoder with conditional boxes to prevent boxes from being randomly placed across the entire image. Third, we propose an aligned decoder for alignment between oriented boxes and image features. The extensive experiments demonstrate ReDiffDet achieves promising performance and significantly outperforms the diffusion-based baseline detector.

## Results and models

DOTA1.0


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 47.69 | 73.88  |   52.17    |   le90   |      2x      |  -  | 4=2gpu*2img/gpu      | [GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dotav1.0.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log]() |

This is your evaluation result for task 1 (VOC metrics):

mAP: 0.7387758407976989

ap of each class: plane:0.8850394908129608, baseball-diamond:0.7901977747706781, bridge:0.5057504797851325, ground-track-field:0.7170557083792797, small-vehicle:0.7852998720627247, large-vehicle:0.8320931918879327, ship:0.8824600057919189, tennis-court:0.9084433118119101, basketball-court:0.8425974293353717, storage-tank:0.8032986565311625, soccer-ball-field:0.5587179317542539, roundabout:0.5704119054786441, harbor:0.7469883448696094, swimming-pool:0.6787017814257459, helicopter:0.5745817272681583

COCO style result:

AP50: 0.7387758407976989

AP75: 0.521672244119724

mAP: 0.47687469541788785



DIOR-R


|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (800,800,200) | 46.34 | 69.80  |   48.16    |   le90   |      2x      |  -  | 4=2gpu*2img/gpu      | [GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior.py](./configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior/20241208_124846/20241208_124846.log) \| [results](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior/20250704_182754/20250704_182754.log) |


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
