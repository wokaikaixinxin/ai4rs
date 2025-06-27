# DiffusionDet: Diffusion Model for Object Detection

> [DiffusionDet: Diffusion Model for Object Detection](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/ShoufaChen/DiffusionDet/raw/main/teaser.png" width="800"/>
</div>


We propose DiffusionDet, a new framework that formulates object detection as a denoising diffusion process from noisy boxes to object boxes. During the training stage, object boxes diffuse from ground-truth boxes to random distribution, and the model learns to reverse this noising process. In inference, the model refines a set of randomly generated boxes to the output results in a progressive way. Our work possesses an appealing property of flexibility, which enables the dynamic number of boxes and iterative evaluation. The extensive experiments on the standard benchmarks show that DiffusionDet achieves favorable performance compared to previous well-established detectors. For example, DiffusionDet achieves 5.3 AP and 4.8 AP gains when evaluated with more boxes and iteration steps, under a zero-shot transfer setting from COCO to CrowdHuman.

## Results and models

DOTA1.0



|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 38.86 | 67.03 | 39.30 | le90 |  2x  |  -  |   4=2gpu*2img/gpu   | [diffdet_r50_b900_step1_stage6_2x_csl_dota1.0](./configs/diffdet_r50_b900_step1_stage6_2x_csl_dota1.0.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Rotated_DiffusionDet/diffdet_r50_b900_step1_stage6_2x_csl_dota1.0/20250626_092920/20250626_092920.log) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Rotated_DiffusionDet/diffdet_r50_b900_step1_stage6_2x_csl_dota1.0/Task1.zip) |

This is your evaluation result for task 1 (VOC metrics):

mAP: 0.6703159019190862

ap of each class: plane:0.8522178462283937, baseball-diamond:0.6960863285725678, bridge:0.40323070163064867, ground-track-field:0.5601069939542106, small-vehicle:0.7262827489788363, large-vehicle:0.740019515240742, ship:0.7852046765499338, tennis-court:0.9011282074515075, basketball-court:0.7966173171392549, storage-tank:0.8129211675837761, soccer-ball-field:0.4335662741656772, roundabout:0.6207016386669268, harbor:0.5223461855695107, swimming-pool:0.614061012702928, helicopter:0.5902479143513809

COCO style result:

AP50: 0.6703159019190862
AP75: 0.39302584590130013
mAP: 0.38856435178712545


## Citation

```
@InProceedings{Chen_2023_ICCV,
    author    = {Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
    title     = {DiffusionDet: Diffusion Model for Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {19830-19843}
}
```
