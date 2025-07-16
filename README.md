<div align="center">
  <img src="resources/ai4rs-logo.png" width="800"/>
</div>



<div align="center">

[📘使用文档](https://mmrotate.readthedocs.io/zh_CN/1.x/) &#124;
[🛠️安装教程](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) &#124;
[👀模型库](https://mmrotate.readthedocs.io/zh_CN/1.x/model_zoo.html) &#124;


[📘Documentation](https://mmrotate.readthedocs.io/en/1.x/) &#124;
[🛠️Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) &#124;
[👀Model Zoo](https://mmrotate.readthedocs.io/en/1.x/model_zoo.html) 

</div>




## Introduction


<!--希望在MMLab基础上，特别是MMDetection、MMRotate的基础上集成遥感相关的工作。-->
We hope to integrate remote sensing related work based on MMLab, especially MMDetection and MMRotate.


## Model Zoo

<details open>
<summary><b>Oriented Object Detection - Architecture </b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [Rotated RetinaNet-OBB/HBB<br>(ICCV'2017)](configs/rotated_retinanet/README.md) | [Rotated FasterRCNN-OBB<br>(TPAMI'2017)](configs/rotated_faster_rcnn/README.md) | [Rotated RepPoints-OBB<br>(ICCV'2019)](configs/rotated_reppoints/README.md) | [Rotated FCOS<br>(ICCV'2019)](configs/rotated_fcos/README.md) |
| [RoI Transformer<br>(CVPR'2019)](configs/roi_trans/README.md) | [Gliding Vertex<br>(TPAMI'2020)](configs/gliding_vertex/README.md) | [Rotated ATSS-OBB<br>(CVPR'2020)](configs/rotated_atss/README.md) |  |
| [R<sup>3</sup>Det<br>(AAAI'2021)](configs/r3det/README.md) | [S<sup>2</sup>A-Net<br>(TGRS'2021)](configs/s2anet/README.md) | [ReDet<br>(CVPR'2021)](configs/redet/README.md) | [Beyond Bounding-Box<br>(CVPR'2021)](configs/cfa/README.md) |
| [Oriented R-CNN<br>(ICCV'2021)](configs/oriented_rcnn/README.md) |  |  | [SASM<br>(AAAI'2022)](configs/sasm_reppoints/README.md) |
| [Oriented RepPoints<br>(CVPR'2022)](configs/oriented_reppoints/README.md) |  |  |  |
| [RTMDet<br>(arXiv)](configs/rotated_rtmdet/README.md) | [Rotated DiffusionDet<br>(ICCV'2023)](./projects/rotated_DiffusionDet/README.md) |  | [OrientedFormer<br>(TGRS' 2024)](projects/OrientedFormer/README.md)|
|[ReDiffDet base<br>(CVPR'2025)](./projects/GSDet_baseline/README_ReDiffDet_baseline.md)|[GSDet base<br>(IJCAI'2025)](./projects/GSDet_baseline/README_GSDet_baseline.md)|||

</details>


<details open>
<summary><b>Oriented Object Detection - Loss</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [GWD<br>(ICML'2021)](configs/gwd/README.md) | [KLD<br>(NeurIPS'2021)](configs/kld/README.md) | [KFIoU<br>(ICLR'2023)](configs/kfiou/README.md) | |
</details>

<details open>
<summary><b>Oriented Object Detection - Coder</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [CSL<br>(ECCV'2020)](configs/csl/README.md) | [Oriented R-CNN<br>(ICCV'2021)](configs/oriented_rcnn/README.md) | [PSC<br>(CVPR'2023)](configs/psc/README.md) | [GauCho<br>(CVPR'2025)](projects/GauCho/README.md) |
</details>


<details open>
<summary><b>Oriented Object Detection - Backbone</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [ConvNeXt<br>(CVPR'2022)](./configs/convnext/README.md)| [LSKNet<br>(ICCV'2023)](projects/LSKNet/README.md)     |   [PKINet<br>(CVPR'2024)](./projects/PKINet/README.md)  |   [SARDet 100K<br>(Nips'2024)](./projects/SARDet_100K/README.md)  |
</details>


<details open>
<summary><b>Oriented Object Detection - Weakly Supervise</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [H2RBox<br>(ICLR'2023)](configs/h2rbox/README.md) | [H2RBox-v2<br>(Nips'2023)](configs/h2rbox_v2/README.md) |  [Point2Rbox<br>(CVPR'2024)](./projects/Point2Rbox/README.md)   |  [Point2Rbox-v2<br>CVPR'2025](./projects/Point2Rbox_v2/README.md)
   |   
</details>

<details open>
<summary><b>SAR</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [SARDet 100K<br>(Nips'2024)](./projects/SARDet_100K/README.md) | [RSAR <br> (CVPR'2025)](./tools/data/rsar/README.md) |   |     |   

<details open>
<summary><b>SAM</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [MMRotate SAM](./projects/mmrotate-sam/README.md) |  |   |     |   

## Installation


<!--请参考[快速入门文档](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)进行安装。-->
<!--Please read the [GET STARTED](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) for installation.-->


To support H2rbox_v2, point2rbox, and mamba, we use **pytorch-2.x**


<!--**第一步：** 安装Anaconda 或 Miniconda-->

**Step 1:** Install Anaconda or Miniconda

<!--**第二步：** 创建一个虚拟环境并且切换至该虚拟环境中-->

**Step 2:** Create a virtual environment

```
conda create --name ai4rs python=3.10 -y
conda activate ai4rs
```

<!--**第三步：** 根据 [Pytorch的官方说明](https://pytorch.org/get-started/previous-versions/) 安装Pytorch, 例如：-->

**Step 3:** Install Pytorch according to [official instructions](https://pytorch.org/get-started/previous-versions/). For example:

```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Verify whether pytorch supports cuda

```
python -c "import torch; print(torch.cuda.is_available())"
```



<!--**第四步：** 安装 MMEngine 和 MMCV, 并且我们建议使用 MIM 来完成安装-->

**Step 4:** Install MMEngine and MMCV, and we recommend using MIM to complete the installation


```
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv>2.0.0rc4, <2.2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<!--**第五步：** 安装 MMDetection-->

**Step 5:** Install MMDetection

```
mim install 'mmdet>3.0.0rc6, <3.4.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<!--**第六步：** 安装 ai4rs-->

**Step 6:** Install ai4rs

```
git clone https://github.com/wokaikaixinxin/ai4rs.git
cd ai4rs
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

**Step 7:** Version of NumPy

If the NumPy version is incompatible, downgrade the NumPy version to 1.x.

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.1 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
```

```
pip install "numpy<2" -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## Data Preparation



Please refer to [data_preparation.md](tools/data/README.md) to prepare the data


|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [DOTA](./tools/data/dota/README.md) | [DIOR](./tools/data/dior/README.md) |  [SSDD](./tools/data/ssdd/README.md) |  [HRSC](./tools/data/hrsc/README.md)   |   
| [HRSID](./tools/data/hrsid/README.md) | [SRSDD](./tools/data/srsdd/README.md) | [RSDD](./tools/data/rsdd/README.md)  |  [ICDAR2015](./tools/data/icdar2015/README.md)   |  
| [SARDet 100K](./tools/data/sardet_100k/README.md) | [RSAR](./tools/data/rsar/README.md) |   |     |    



## Train

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

## Test

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

## Getting Started

<!--请阅读[概述](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)对 Openmmlab 进行初步的了解。-->

Please see [Overview](https://mmrotate.readthedocs.io/en/1.x/overview.html) for the general introduction of Openmmlab.

<!--为了帮助用户更进一步了解 Openmmlab，我们准备了用户指南和进阶指南，请阅读我们的[文档](https://mmrotate.readthedocs.io/zh_CN/1.x/)：-->

For detailed user guides and advanced guides, please refer to our [documentation](https://mmrotate.readthedocs.io/en/1.x/):


## FAQ

<!--请参考 [FAQ](docs/en/notes/faq.md) 了解其他用户的常见问题。-->

Please refer to [FAQ](https://github.com/open-mmlab/mmrotate/blob/1.x/docs/en/notes/faq.md) for frequently asked questions.




## Acknowledgement

[OpenMMLab](https://openmmlab.com)

[OpenMMLab platform](https://platform.openmmlab.com)

[MMDetection](https://github.com/open-mmlab/mmdetection)

[MMRotate](https://github.com/open-mmlab/MMRotate)

## Citation

<!--如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 ai4rs-->

If you use this toolbox or benchmark in your research, please cite this project ai4rs

```bibtex

```



