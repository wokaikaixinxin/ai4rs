# VLM4RSDet: Collaborative Optimization with Vision-Language Model for Enhancing Remote Sensing Object Detection (CVPR2026)


[CVPR official paper link](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_VLM4RSDet_Collaborative_Optimization_with_Vision-Language_Model_for_Enhancing_Remote_Sensing_CVPR_2026_paper.html)


[Official github repo](https://github.com/cszzshi/VLM4RSDet)

<div align=center>
<img src="https://github.com/cszzshi/VLM4RSDet/blob/main/network.png" width="800"/>
</div>

## Model Zoo

[🤗Faster R-CNN w/ VLM4RSDet on VisDrone](https://huggingface.co/cszzshi/VLM4RSDet).

Download
```Shell
export HF_ENDPOINT="https://hf-mirror.com"
# pip install -U huggingface_hub
huggingface-cli download cszzshi/VLM4RSDet \
    --local-dir work_dirs/VLM4RSDet/visdrone/faster-rcnn_r50_fpn_1x_visdrone
```

Rename checkpoint

```Shell
cd work_dirs/VLM4RSDet/visdrone/faster-rcnn_r50_fpn_1x_visdrone
mv Faster\ R-CNN\ w\ VLM4RSDet\ on\ VisDrone\ epoch_12.pth epoch_12.pth
cd ../../../../
```

## Dataset

The official download link of VisDrone dataset.

 - trainset (1.44 GB): [Baidu Yun](https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A) | [Google Drive](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing)

 - valset (0.07 GB): [Baidu Yun](https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g) | [Google Drive](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing)

```Shell
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── visdrone
│   │   ├── val
│   │   │   ├── images (548 jpg)
│   │   │   ├── annotations (548 txt)
│   │   ├── train
│   │   │   ├── images ()
│   │   │   ├── annotations ()
```

Convert visdrone to coco format.

train set
```Shell
python projects/VLM4RSDet/vlm4rsdet/visdrone2coco.py \
    --image_dir data/visdrone/train/images \
    --txt_dir data/visdrone/train/annotations \
    --save_json data/visdrone/train/annotations/result.json
```

val set
```Shell
python projects/VLM4RSDet/vlm4rsdet/visdrone2coco.py \
    --image_dir data/visdrone/val/images \
    --txt_dir data/visdrone/val/annotations \
    --save_json data/visdrone/val/annotations/result.json
```


## Evaluation

Test the trained weight using 4 GPUs.

```Shell 
CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh projects/VLM4RSDet/configs/faster-rcnn_r50_fpn_1x_visdrone.py work_dirs/VLM4RSDet/visdrone/faster-rcnn_r50_fpn_1x_visdrone/epoch_12.pth 4
```

Test the trained weight using a single GPU.

```Shell 
python tools/test.py projects/VLM4RSDet/configs/faster-rcnn_r50_fpn_1x_visdrone.py work_dirs/VLM4RSDet/visdrone/faster-rcnn_r50_fpn_1x_visdrone/epoch_12.pth
```

**Official github repo only support test.**



## Citation 
```
@InProceedings{Shi_2026_CVPR,
    author = {Shi, Shuohao and Fang, Qiang and Xu, Xin},
    title = {VLM4RSDet: Collaborative Optimization with Vision-Language Model for Enhancing Remote Sensing Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2026},
    pages = {18450-18460}
}
```