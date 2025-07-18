# Adaptive Rotated Convolution for Rotated Object Detection 

> [Adaptive Rotated Convolution for Rotated Object Detection](https://openaccess.thecvf.com/content/ICCV2023/html/Pu_Adaptive_Rotated_Convolution_for_Rotated_Object_Detection_ICCV_2023_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/LeapLabTHU/ARC/raw/main/figs/module.png" width="800"/>
</div>

Rotated object detection aims to identify and locate objects in images with arbitrary orientation. In this scenario, the oriented directions of objects vary considerably across different images, while multiple orientations of objects exist within an image. This intrinsic characteristic makes it challenging for standard backbone networks to extract high-quality features of these arbitrarily orientated objects. In this paper, we present Adaptive Rotated Convolution (ARC) module to handle the aforementioned challenges. In our ARC module, the convolution kernels rotate adaptively to extract object features with varying orientations in different images, and an efficient conditional computation mechanism is introduced to accommodate the large orientation variations of objects within an image. The two designs work seamlessly in rotated object detection problem. Moreover, ARC can conveniently serve as a plug-and-play module in various vision backbones to boost their representation ability to detect oriented objects accurately. Experiments on commonly used benchmarks (DOTA and HRSC2016) demonstrate that equipped with our proposed ARC module in the backbone network, the performance of multiple popular oriented object detectors is significantly improved (e.g. +3.03% mAP on Rotated RetinaNet and +4.16% on CFA). Combined with the highly competitive method Oriented R-CNN, the proposed approach achieves state-of-the-art performance on the DOTA dataset with 81.77% mAP.


## Results and models


**DOTA-v1.0**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | lr | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :---: | :--------: | :---------------------------------------------: | :-------------------------------: |
| ARC <br> (1024,1024,200) | 46.36 | 76.74  |  49.00  |   le90   |  1x  | -  | 5e-3 | 2=1gpu*<br>2img/gpu      | [oriented-rcnn-le90_arc_<br>r50_fpn_1x_dota.py](./configs/oriented-rcnn-le90_arc_r50_fpn_1x_dota.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ARC/oriented-rcnn-le90_arc_r50_fpn_1x_dota/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ARC/oriented-rcnn-le90_arc_r50_fpn_1x_dota/20250717_215756/20250717_215756.log) \| <br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/ARC/oriented-rcnn-le90_arc_r50_fpn_1x_dota/Task1.zip)|

Note: This is the **unofficial** checkpoint. The official code is [here](https://github.com/LeapLabTHU/ARC).  
Note: The official result is **77.35 AP50** on DOTA-v1.0, but in this project the result is **76.74 AP50** on DOTA-v1.0.

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7674185176745701  
ap of each class: plane:0.8913062699625618, baseball-diamond:0.8151082772971205, bridge:0.5479157228649669, ground-track-field:0.723641256930776, small-vehicle:0.790871270656151, large-vehicle:0.8407983853616632, ship:0.8805662786083926, tennis-court:0.9080753236768999, basketball-court:0.856818820298745, storage-tank:0.8506394734480863, soccer-ball-field:0.6441181531452128, roundabout:0.6612835149703274, harbor:0.7326476161965203, swimming-pool:0.6996791929233002, helicopter:0.6678082087778289
COCO style result:  
AP50: 0.7674185176745701  
AP75: 0.4899630195117957  
mAP: 0.4635887436735895  


**Train**

```
python tools/train.py config_path
``` 

For example:

```
python tools/train.py projects/ARC/configs/oriented-rcnn-le90_arc_r50_fpn_1x_dota.py
```


**Test**
```
python tools/test.py config_path checkpoint_path
```  

For example:

```
python tools/test.py projects/ARC/configs/oriented-rcnn-le90_arc_r50_fpn_1x_dota.py work_dirs/oriented-rcnn-le90_arc_r50_fpn_1x_dota/epoch_12.pth
```


## Citation

```
@InProceedings{Pu_2023_ICCV,
    author    = {Pu, Yifan and Wang, Yiru and Xia, Zhuofan and Han, Yizeng and Wang, Yulin and Gan, Weihao and Wang, Zidong and Song, Shiji and Huang, Gao},
    title     = {Adaptive Rotated Convolution for Rotated Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {6589-6600}
}
```
