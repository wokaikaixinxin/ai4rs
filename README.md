<div align="center">
  <img src="resources/ai4rs-logo.png" width="800"/>
</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=wokaikaixinxin/ai4rs)
[![GitHub Repo stars](https://img.shields.io/github/stars/wokaikaixinxin/ai4rs?style=social)](https://github.com/wokaikaixinxin/ai4rs/stargazers)

</div>


<div align="center">

[📘使用文档](https://mmrotate.readthedocs.io/zh_CN/1.x/) &#124;
[🛠️安装教程](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) &#124;
[👀模型库](https://mmrotate.readthedocs.io/zh_CN/1.x/model_zoo.html) &#124;


[📘Documentation](https://mmrotate.readthedocs.io/en/1.x/) &#124;
[🛠️Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) &#124;
[👀Model Zoo](https://mmrotate.readthedocs.io/en/1.x/model_zoo.html) 

</div>




## Introduction 👋


We hope to integrate remote sensing related work based on **MMLab**, especially **MMDetection** and **MMRotate**.


## Model Zoo 🐅

<details open>
<summary><b>Real Time </b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
|[Rotated YOLOX (arXiv 2021)](./projects/rotated_yolox/README.md) |[RTMDet (arXiv 2022)](configs/rotated_rtmdet/README.md)  |  [Rotated YOLOMS (TPAMI 2025)](./projects/rotated_yoloms/README.md)    |  [RTDETR (CVPR' 2024)](./projects/rtdetr/README.md)   |
|  [O2-RTDETR (TGRS 2026)](./projects/rotated_rtdetr/README.md)   | [rtdetr-mmdet](https://github.com/flytocc/rtdetr-mmdet) |     |     |

</details>


<details open>
<summary><b>Oriented Object Detection - Architecture </b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [Rotated RetinaNet-OBB/HBB<br>(ICCV'2017)](configs/rotated_retinanet/README.md) | [Rotated FasterRCNN-OBB<br>(TPAMI'2017)](configs/rotated_faster_rcnn/README.md) | [Rotated RepPoints-OBB<br>(ICCV'2019)](configs/rotated_reppoints/README.md) | [Rotated FCOS<br>(ICCV'2019)](configs/rotated_fcos/README.md) |
| [RoI Transformer<br>(CVPR'2019)](configs/roi_trans/README.md) | [Gliding Vertex<br>(TPAMI'2020)](configs/gliding_vertex/README.md) | [Rotated ATSS-OBB<br>(CVPR'2020)](configs/rotated_atss/README.md) | [R<sup>3</sup>Det<br>(AAAI'2021)](configs/r3det/README.md) |
 | [S<sup>2</sup>A-Net<br>(TGRS'2021)](configs/s2anet/README.md) | [ReDet<br>(CVPR'2021)](configs/redet/README.md) | [Beyond Bounding-Box<br>(CVPR'2021)](configs/cfa/README.md) | [Oriented R-CNN<br>(ICCV'2021)](configs/oriented_rcnn/README.md) | 
| [Rotated YOLOX <br>(arXiv 2021)](./projects/rotated_yolox/README.md) | [Rotated Deformable DETR <br> (ICLR'2021)](./projects/rotated_deformable_detr/README.md) |[SASM<br>(AAAI'2022)](configs/sasm_reppoints/README.md) | [Oriented RepPoints<br>(CVPR'2022)](configs/oriented_reppoints/README.md) |  
| [RTMDet<br>(arXiv 2022)](configs/rotated_rtmdet/README.md) |[Rotated DiffusionDet<br>(ICCV'2023)](./projects/rotated_DiffusionDet/README.md) | [Rotated DINO<br>(ICLR 2023)](./projects/rotated_dino/README.md)  | [OrientedFormer<br>(TGRS' 2024)](projects/OrientedFormer/README.md) | 
| [RTDETR<br> (CVPR' 2024)](./projects/rtdetr/README.md) | [ReDiffDet base<br>(CVPR'2025)](./projects/GSDet_baseline/README_ReDiffDet_baseline.md)| [GSDet base<br>(IJCAI'2025)](./projects/GSDet_baseline/README_GSDet_baseline.md)|  [Rotated YOLOMS<br>(TPAMI 2025)](./projects/rotated_yoloms/README.md) |
|  [MessDet<br>(ICCV'2025)](./projects/MessDet/README.md)   | [AMMBA<br>(TGRS'2025)](./projects/AMMBA/README.md) | [RHINO<br>(WACV'2025)](./projects/RHINO/README.md) |  [RQFormer<br>(ESWA'2025)](https://github.com/wokaikaixinxin/RQFormer) |
| [HERO <br>(AAAI'2026)](./projects/HERO/README.md) | [O2-RTDETR<br>(TGRS' 2026)](./projects/rotated_rtdetr/README.md)  |     |     |
</details>


<details open>
<summary><b>Oriented Object Detection - Loss</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [GWD<br>(ICML'2021)](configs/gwd/README.md) | [KLD<br>(NeurIPS'2021)](configs/kld/README.md) | [KFIoU<br>(ICLR'2023)](configs/kfiou/README.md) | |
</details>

<details open>
<summary><b>Oriented Object Detection - Coder</b></summary>

|     |     |     |     |     |
| :---: | :---: | :---: | :---: | :---: |
| [CSL<br>(ECCV'2020)](configs/csl/README.md) | [Oriented R-CNN<br>(ICCV'2021)](configs/oriented_rcnn/README.md) | [PSC<br>(CVPR'2023)](configs/psc/README.md) | [ACM<br>(CVPR'2024)](./projects/ACM/README.md) |[GauCho<br>(CVPR'2025)](projects/GauCho/README.md) |
</details>


<details open>
<summary><b>Oriented Object Detection - Backbone</b></summary>

|       |       |       |       |       |
| :---: | :---: | :---: | :---: | :---: |
| [ConvNeXt<br>(CVPR'2022)](./configs/convnext/README.md)| [LSKNet<br>(ICCV'2023)](projects/LSKNet/README.md)  | [ARC<br>(ICCV'2023)](./projects/ARC/README.md)   |   [PKINet<br>(CVPR'2024)](./projects/PKINet/README.md)  |   [SARDet 100K<br>(Nips'2024)](./projects/SARDet_100K/README.md)  | 
| [GRA<br>(ECCV'2024)](./projects/GRA/README.md) | [LEGNet<br>(ICCVW'2025)](./projects/LEGNet/README.md) | [Strip R-CNN<br>(AAAI'2026)](./projects/Strip_RCNN/README.md)   |  [LWGANet<br>(AAAI'2026)](./projects/LWGANet/README.md)    |  [Fourier Angle Align<br>(CVPR'2026)](./projects/FAA/README.md)     |       
| [PKINet-v2<br>(ECCV'2026)](./projects/PKINet_v2/README.md)      |       |       |       |       |
</details>


<details open>
<summary><b>Oriented Object Detection - Weakly Supervise</b></summary>

|       |       |       |       |       |
| :---: | :---: | :---: | :---: | :---: |
| [H2RBox<br>(ICLR'2023)](configs/h2rbox/README.md) | [H2RBox-v2<br>(Nips'2023)](configs/h2rbox_v2/README.md) |  [Point2Rbox<br>(CVPR'2024)](./projects/Point2Rbox/README.md)   |  [Point2Rbox-v2<br>(CVPR'2025)](./projects/Point2Rbox_v2/README.md)| [WhollyWOOD<br>(TPAMI'2025)](./projects/WhollyWOOD/README.md)   |
| [ABBSPO<br>(CVPR'2025)](./projects/ABBSPO/README.md)      |       |       |       |       |
</details>


<details open>
<summary><b>Oriented Object Detection - Semi Supervise</b></summary>

Coming soon
</details>


<details open>
<summary><b>SAR</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [SARDet 100K (Nips'2024)](./projects/SARDet_100K/README.md) | [RSAR (CVPR'2025)](./tools/data/rsar/README.md) |   |     |   
</details>


<details open>
<summary><b>SAM</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [SAM (ICCV'2023)](./projects/segment_anything/README.md) | [MMRotate SAM (git 2023)](./projects/mmrotate-sam/README.md) |   [Grounded SAM (Arxiv'2024)](./projects/grounded_sam/README.md) | [SAM2 (Arxiv'2024)](./projects/sam2/README.md) |
| [SAM2 ai4rs (git'2026)](./projects/sam2_ai4rs/README.md) | [SAM2.1 (git'2024)](./projects/sam2/README.md) | [SAM2.1 ai4rs (git'2026)](./projects/sam2_1_ai4rs/README.md) | Grounded SAM2 |
 | [SAM3 (Arxiv'2025)](./projects/sam3/README.md) | [SegEarth-ov3 (Arxiv'2025)](./projects/SegEarth_OV_3/README.md)  | | |
</details>


<details open>
<summary><b>Scene graph generation (SGG)</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [STAR (TPAMI'2025)](./tools/data/star/README.md) | [ReCon1M (TGRS'2025)](./tools/data/recon1m/README.md) |   |     |  
</details>


<details open>
<summary><b>Change Detection</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [FC-EF (ICIP'2018)](./projects/fcsn/README.md) | [FC-Siam-conc (ICIP'2018)](./projects/fcsn/README.md) | [FC-Siam-diff (ICIP'2018)](./projects/fcsn/README.md) |   [Changer (TGRS'2023)](./projects/changer/README.md)  |  
|  [BiT (TGRS' 2021)](./projects/bit/README.md)   |  [IDA-SiamNet (TGRS' 2024)](./projects/idasiamnet/README.md)   |     |     |
</details>


<details open>
<summary><b>Instance Segmentation</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [Mask RCNN (ICCV'2017)](./projects/mask_rcnn/README.md) |  [CATNet (TNNLS'2024)](./projects/CATNet/README.md)   |     |     |
</details>

<details open>
<summary><b>Semantic Segmentation</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [SegEarth-OV3 (Arxiv'2025)](./projects/SegEarth_OV_3/README.md) |     |     |     |
</details>


<details open>
<summary><b>RGB Infrared</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
|  [DroneVehicle (CSVT'22)](./tools/data/dronevehicle/README.md)   |     |     |     |
</details>


## Installation ⚙️ [![Bilibili](https://img.shields.io/badge/Installation_Tutorial-fb7299?style=flat-square&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1Ufw4zyEhR/)


To support H2rbox_v2, point2rbox, and mamba, we use **pytorch-2.x**.


**Step 1:** Install Anaconda or Miniconda


**Step 2:** Create a virtual environment

```
conda create --name ai4rs python=3.10 -y
conda activate ai4rs
```


**Step 3:** Install Pytorch according to [official instructions](https://pytorch.org/get-started/previous-versions/). For example:

```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify whether pytorch supports cuda

```
python -c "import torch; print(torch.cuda.is_available())"
```


**Step 4:** Install MMEngine and MMCV, and we recommend using MIM to complete the installation


```
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv==2.2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```


**Step 5:** Install MMDetection and MMSegmentation(for change detection)

```
mim install 'mmdet>3.0.0rc6, <3.4.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmsegmentation>=1.2.2" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Change the version number in the `mmdet` code.
```
python -c "import mmdet; print(mmdet.__file__)"
# ...
#   File "/root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmdet/__init__.py", line 17, 
#     in <module> and mmcv_version < digit_version(mmcv_maximum_version)), \
# AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0.
```

 modify `/root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmdet/__init__.py`
```
vim '/root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmdet/__init__.py'
# mmcv_maximum_version = '2.2.0' -> '2.3.0'
```

Change the version number in the `mmseg` code.
```
python -c "import mmseg; print(mmseg.__file__)"
# ...
#   File "/root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmseg/__init__.py", line 61,
#      in <module> assert (mmcv_min_version <= mmcv_version < mmcv_max_version), \
# AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4.
```

 modify `/root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmseg/__init__.py`
```
vim '/root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmseg/__init__.py'
# MMCV_MAX = '2.2.0' -> '2.3.0'
```


**Step 6:** Install ai4rs

```
git clone https://github.com/wokaikaixinxin/ai4rs.git
cd ai4rs
```

Option 1: Basic Installation  
Use this for a lightweight setup with only the core functional dependencies.  
```
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Option 2: Full Installation (Recommended)  
Use this if you want all features and dependencies (e.g., change detection). 
```
pip install -v -e .[all] -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## Data Preparation 🗃️



Please refer to [data_preparation.md](tools/data/README.md) to prepare the data


|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [DOTA (CVPR'2018)](./tools/data/dota/README.md) | [DIOR (TGRS'2022)](./tools/data/dior/README.md) |  [SSDD (RS'2021)](./tools/data/ssdd/README.md) |  [HRSC (ICPRAM'2017)](./tools/data/hrsc/README.md)   |   
| [HRSID (Access'2020)](./tools/data/hrsid/README.md) | [SRSDD (RS'2021)](./tools/data/srsdd/README.md) | [RSDD (JR'2022)](./tools/data/rsdd/README.md)  |  [ICDAR2015 (ICDAR'2015)](./tools/data/icdar2015/README.md)   |  
| [SARDet 100K (Nips'2024)](./tools/data/sardet_100k/README.md) | [RSAR (CVPR'2025)](./tools/data/rsar/README.md) | [FAIR1M (ISPRS'2022)](./tools/data/fair/README.md)  | [STAR (TPAMI'2025)](./tools/data/star/README.md)    |    
| [ReCon1M (TGRS'2025)](./tools/data/recon1m/README.md)    |  [CODrone (Arxiv'2025)](./tools/data/codrone/README.md)   | [KFGOD (RS'2025)](./tools/data/kfgod/README.md)  | [LEVIR-CD (TGRS'2020)](./tools/data/levir_cd/README.md)    |
| [iSAID (CVPRW'2019)](./tools/data/isaid/README.md) |  [DroneVehicle (CSVT'22)](./tools/data/dronevehicle/README.md)   |  [NWPU (ISPRS'2014)](./tools/data/nwpu/README.md)   |     | 


## Train 📈

**Single-node single-GPU**  
```
python tools/train.py config_path
```  
For example:  
```
python tools/train.py projects/GSDet_baseline/configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior.py
```

**Single-node multi-GPU**  
```
bash tools/dist_train.sh config_path num_gpus
```  
For example:  
```
bash tools/dist_train.sh projects/GSDet_baseline/configs/GSDet_r50_b900_h2h4_h2r1_r2r1_2x_dior.py 2
```

## Test 🧪

**Single-node single-GPU**  
```
python tools/test.py config_path checkpoint_path
```  
For example:  
```
python tools/test.py configs/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dota.py work_dirs/h2rbox_v2-le90_r50_fpn-1x_dota-fa5ad1d2.pth
```

**Single-node multi-GPU**  
```
bash tools/dist_test.sh config_path checkpoint_path num_gpus
```  
For example:  
```
bash tools/dist_test.sh configs/h2rbox_v2/h2rbox_v2-le90_r50_fpn-1x_dota.py work_dirs/h2rbox_v2-le90_r50_fpn-1x_dota-fa5ad1d2.pth 2
```


## Getting Started 🚀


Please see [Overview](https://mmrotate.readthedocs.io/en/1.x/overview.html) for the general introduction of Openmmlab.


For detailed user guides and advanced guides, please refer to our [documentation](https://mmrotate.readthedocs.io/en/1.x/):


## FAQ 🤔


Please refer to [FAQ](https://github.com/open-mmlab/mmrotate/blob/1.x/docs/en/notes/faq.md) for frequently asked questions.





## Acknowledgement 🙏

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [OpenMMLab](https://openmmlab.com) | [OpenMMLab platform](https://platform.openmmlab.com) |  [MMDetection](https://github.com/open-mmlab/mmdetection)   | [MMRotate](https://github.com/open-mmlab/MMRotate) |
|  [open-cd](https://github.com/likyoo/open-cd) |   [rtdetr-mmdet](https://github.com/flytocc/rtdetr-mmdet)  |     |     |










## Citation 🌟


If you use this toolbox or benchmark in your research, please cite this project ai4rs

```bibtex

```


<div align="center">
⭐ <b>If you find this project helpful, please give us a star!</b> ⭐
</div>
