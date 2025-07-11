# Point2RBox

> [SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection](https://arxiv.org/abs/2403.06534)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/zcablii/SARDet_100K/raw/main/MSFA/docs/SARDet100K_samples.png" width="800"/>
</div>

Synthetic Aperture Radar (SAR) object detection has gained significant attention recently due to its irreplaceable all-weather imaging capabilities. However, this research field suffers from both limited public datasets (mostly comprising <2K images with only mono-category objects) and inaccessible source code. To tackle these challenges, we establish a new benchmark dataset and an open-source method for large-scale SAR object detection. Our dataset, SARDet-100K, is a result of intense surveying, collecting, and standardizing 10 existing SAR detection datasets, providing a large-scale and diverse dataset for research purposes. To the best of our knowledge, SARDet-100K is the first COCO-level large-scale multi-class SAR object detection dataset ever created. With this high-quality dataset, we conducted comprehensive experiments and uncovered a crucial challenge in SAR object detection: the substantial disparities between the pretraining on RGB datasets and finetuning on SAR datasets in terms of both data domain and model structure. To bridge these gaps, we propose a novel Multi-Stage with Filter Augmentation (MSFA) pretraining framework that tackles the problems from the perspective of data input, domain transition, and model migration. The proposed MSFA method significantly enhances the performance of SAR object detection models while demonstrating exceptional generalizability and flexibility across diverse models. This work aims to pave the way for further advancements in SAR object detection.

## Results and models

**SARDet_100K -- Faster RCNN**

|Framework |   Backbone  |Pretrain|  lr schd | mAP | AP50 | AP75 |  Batch Size |                       Configs                       |                                                                                                                    Download                                                                                                                    |
| :--------------: | :--------------: | :--------------: | :---: | :-----:  | :------------: | :-: | :--------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|Faster RCNN| ResNet50 | MSFA | 1x |   51.1    |  83.9   | 54.7  | 16=8gpu*2img/gpu  | [fg_frcnn_dota_pretrain_sar_wavelet_r50.py](./configs/r50_dota_pretrain/fg_frcnn_dota_pretrain_sar_wavelet_r50.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/MSFA/fg_frcnn_dota_pretrain_sar_wavelet_r50/best_coco_bbox_mAP_epoch_12.pth)   \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/MSFA/fg_frcnn_dota_pretrain_sar_wavelet_r50/20240116_033917/20240116_033917.log)\| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/MSFA/fg_frcnn_dota_pretrain_sar_wavelet_r50/20240116_075728/20240116_075728.log) |


Train
```
bash tools/dist_train.sh projects/SARDet_100K/configs/r50_dota_pretrain/fg_frcnn_dota_pretrain_sar_wavelet_r50.py 2
```  

Test
```
bash tools/dist_test.sh projects/SARDet_100K/configs/r50_dota_pretrain/fg_frcnn_dota_pretrain_sar_wavelet_r50.py work_dirs/fg_frcnn_dota_pretrain_sar_wavelet_r50/best_coco_bbox_mAP_epoch_12.pth 2
```

**SARDet_100K -- Sparse RCNN**

|Framework |   Backbone  |Pretrain|  lr schd | mAP | AP50 | AP75 |  Batch Size |                       Configs                       |                                                                                                                    Download                                                                                                                    |
| :--------------: | :--------------: | :--------------: | :---: | :-----:  | :------------: | :-: | :--------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|Sparse RCNN| ResNet50 | MSFA | 1x |   41.4    |  74.1   | 41.8  | 16=8gpu*2img/gpu  | [sparse-rcnn_r50_dota_pretrain_sar_wavelet.py](./configs/other_detectors/sparse-rcnn_r50_dota_pretrain_sar_wavelet.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/MSFA/sparse-rcnn_r50_dota_pretrain_sar_wavelet/best_coco_bbox_mAP_epoch_11.pth)   \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/MSFA/sparse-rcnn_r50_dota_pretrain_sar_wavelet/20240205_024841/20240205_024841.log)\| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/MSFA/sparse-rcnn_r50_dota_pretrain_sar_wavelet/20240205_093217/20240205_093217.log) |


Train
```
bash tools/dist_train.sh projects/SARDet_100K/configs/other_detectors/sparse-rcnn_r50_dota_pretrain_sar_wavelet.py 2
```

Test
```
bash tools/dist_test.sh projects/SARDet_100K/configs/other_detectors/sparse-rcnn_r50_dota_pretrain_sar_wavelet.py work_dirs/sparse-rcnn_r50_dota_pretrain_sar_wavelet/best_coco_bbox_mAP_epoch_11.pth 2
```



## Citation

```
@inproceedings{li2024sardet100k,
	title={SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection}, 
	author={Yuxuan Li and Xiang Li and Weijie Li and Qibin Hou and Li Liu and Ming-Ming Cheng and Jian Yang},
	year={2024},
	booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
}
```