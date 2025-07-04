# GauCho: Gaussian Distributions with Cholesky Decomposition for Oriented Object Detection

> [GauCho: Gaussian Distributions with Cholesky Decomposition for Oriented Object Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Marques_GauCho_Gaussian_Distributions_with_Cholesky_Decomposition_for_Oriented_Object_Detection_CVPR_2025_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/jhlmarques/GauCho/blob/main/images/concept_gaucho.png" width="800"/>
</div>

Oriented Object Detection (OOD) has received increased attention in the past years, being a suitable solution for detecting elongated objects in remote sensing analysis. In particular, using regression loss functions based on Gaussian distributions has become attractive since they yield simple and differentiable terms. However, existing solutions are still based on regression heads that produce Oriented Bounding Boxes (OBBs), and the known problem of angular boundary discontinuity persists. In this work, we propose a regression head for OOD that directly produces Gaussian distributions based on the Cholesky matrix decomposition. The proposed head, named GauCho, theoretically mitigates the boundary discontinuity problem and is fully compatible with recent Gaussian-based regression loss functions. Furthermore, we advocate using Oriented Ellipses (OEs) to represent oriented objects, which relates to GauCho through a bijective function and alleviates the encoding ambiguity problem for circular objects. Our experimental results show that GauCho can be a viable alternative to the traditional OBB head, achieving results comparable to or better than state-of-the-art detectors for the challenging dataset DOTA.

## Results and models

DOTA1.0

NOTE: This is an unofficial implementation and result. [Official code link](https://github.com/jhlmarques/GauCho).

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 40.46 | 72.46  |   37.03    |   le90   |      1x      |  -  |     2      | [gaussian_fcos_r50_fpn_gaucho<br>_probiou_1x_dota_le90](./configs/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7245925043296044  
ap of each class:   
plane:0.8855710467600557, baseball-diamond:0.7668860091671282, bridge:0.4986973831702632, ground-track-field:0.6159312127648029, small-vehicle:0.80005083972435, large-vehicle:0.7958693676902324, ship:0.8745895973212595, tennis-court:0.9088193360038022, basketball-court:0.83238085820412, storage-tank:0.8419372488797484, soccer-ball-field:0.5497710472382763, roundabout:0.6485814810423159, harbor:0.6411760137182135, swimming-pool:0.6952875542429233, helicopter:0.513338569016573  
COCO style result:  
AP50: 0.7245925043296044  
AP75: 0.3703162110556219  
mAP: 0.4045928928108746


## Citation

```
@InProceedings{Marques_2025_CVPR,
    author    = {Marques, Jos\'e Henrique Lima and Murrugarra-Llerena, Jeffri and Jung, Claudio R.},
    title     = {GauCho: Gaussian Distributions with Cholesky Decomposition for Oriented Object Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3593-3602}
}
```
