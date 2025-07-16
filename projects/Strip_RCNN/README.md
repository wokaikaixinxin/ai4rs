# Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection

> [Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection](https://arxiv.org/abs/2501.03775)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/HVision-NKU/Strip-R-CNN/raw/main/DotaStatis.png" width="800"/>
</div>

While witnessed with rapid development, remote sensing object detection remains challenging for detecting high aspect ratio objects. This paper shows that large strip convolutions are good feature representation learners for remote sensing object detection and can detect objects of various aspect ratios well. Based on large strip convolutions, we build a new network architecture called Strip R-CNN, which is simple, efficient, and powerful. Unlike recent remote sensing object detectors that leverage large-kernel convolutions with square shapes, our Strip R-CNN takes advantage of sequential orthogonal large strip convolutions in our backbone network StripNet to capture spatial information. In addition, we improve the localization capability of remote-sensing object detectors by decoupling the detection heads and equipping the localization branch with strip convolutions in our strip head. Extensive experiments on several benchmarks, for example DOTA, FAIR1M, HRSC2016, and DIOR, show that our Strip R-CNN can greatly improve previous work. In particular, our 30M model achieves 82.75% mAP on DOTA-v1.0, setting a new state-of-the-art record.

## Results and models


**DOTA-v1.0**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 46.26 | 75.79  |  49.64  |   le90   |  1x  |  -  | 8=8gpu*<br>1img/gpu      | [strip_rcnn_t_fpn_1x_dota_le90.py](./configs/strip_rcnn_t_fpn_1x_dota_le90.py) | [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/StripRCNN/strip_rcnn_t_fpn_1x_dota_le90/epoch_12.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/StripRCNN/strip_rcnn_t_fpn_1x_dota_le90/20250716_111957/20250716_111957.log) \| [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/StripRCNN/strip_rcnn_t_fpn_1x_dota_le90/Task1.zip)|

Note: This is the **unofficial** checkpoint. The official code is [here](https://github.com/HVision-NKU/Strip-R-CNN).

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.7578886933949137  
ap of each class: plane:0.8900963130910334, baseball-diamond:0.8252327642545075, bridge:0.5068483104818872, ground-track-field:0.7404200909736047, small-vehicle:0.789040239256087, large-vehicle:0.8398600050940846, ship:0.8816605900794297, tennis-court:0.9054734065066338, basketball-court:0.8594636810327425, storage-tank:0.8486977858429885, soccer-ball-field:0.5696574395594765, roundabout:0.60749533266713, harbor:0.7519148281638536, swimming-pool:0.7061258130639644, helicopter:0.6463438008562831  
COCO style result:  
AP50: 0.7578886933949137  
AP75: 0.4964419758946793  
mAP: 0.46260902296838535  

**Train**

```
bash tools/dist_train.sh config_path num_gpus
``` 

For example:

```
# DOTA-v1.0
bash tools/dist_train.sh projects/Strip_RCNN/configs/strip_rcnn_t_fpn_1x_dota_le90.py 2
```

**Test**
```
python tools/test.py config_path checkpoint_path
```  

For example:

```
# DOTA-v1.0
bash tools/dist_test.sh projects/Strip_RCNN/configs/strip_rcnn_t_fpn_1x_dota_le90.py work_dirs/strip_rcnn_t_fpn_1x_dota_le90/epoch_12.pth 2
```

## Citation

```
@article{yuan2025strip,
  title={Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection},
  author={Yuan, Xinbin and Zheng, ZhaoHui and Li, Yuxuan and Liu, Xialei and Liu, Li and Li, Xiang and Hou, Qibin and Cheng, Ming-Ming},
  journal={arXiv preprint arXiv:2501.03775},
  year={2025}
}
```
