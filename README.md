<div align="center">
  <img src="resources/ai4rs-logo.png" width="800"/>
</div>



<div align="center">

[📘使用文档](https://mmrotate.readthedocs.io/zh_CN/1.x/) &#124;
[🛠️安装教程](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) &#124;
[👀模型库](https://mmrotate.readthedocs.io/zh_CN/1.x/model_zoo.html) &#124;


[📘Documentation](https://mmrotate.readthedocs.io/en/1.x/) &#124;
[🛠️Installation](https://mmrotate.readthedocs.io/en/1.x/install.html) &#124;
[👀Model Zoo](https://mmrotate.readthedocs.io/en/1.x/model_zoo.html) 

</div>




## 介绍 Introduction

AI for Remote Sensing 是一款基于 PyTorch 的人工智能与遥感结合的开源工具箱。


人工智能发展很快，相关工作很多。希望在MMLab基础上，特别是MMDetection、MMRotate的基础上集成遥感相关的工作。

AI for Remote Sensing is an open source toolbox based on PyTorch that combines artificial intelligence and remote sensing.

Artificial intelligence is developing very fast, and there are many related works. We hope to integrate remote sensing related work based on MMLab, especially MMDetection and MMRotate.



## 最新进展 What's New

### 亮点






## 模型库 Model Zoo

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
| [ConvNeXt<br>(CVPR'2022)](./configs/convnext/README.md)| [LSKNet<br>(ICCV'2023)](projects/LSKNet/README.md)     |   [PKINet<br>(CVPR'2024)](./projects/PKINet/README.md)  |     |
</details>


<details open>
<summary><b>Oriented Object Detection - Weakly Supervise</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [H2RBox<br>(ICLR'2023)](configs/h2rbox/README.md) | [H2RBox-v2<br>(Nips'2023)](configs/h2rbox_v2/README.md) |     |     |   
</details>



## 安装 Installation


请参考[快速入门文档](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)进行安装。


**第一步：** 安装Anaconda 或 Miniconda

1： Install Anaconda or Miniconda

**第二步：** 创建一个虚拟环境并且切换至该虚拟环境中

2: Create a virtual environment

```
conda create --name ai4rs python=3.8 -y
conda activate ai4rs
```

**第三步：** 根据 [Pytorch的官方说明](https://pytorch.org/get-started/previous-versions/) 安装Pytorch, 例如：

3: Install Pytorch according to [official instructions](https://pytorch.org/get-started/previous-versions/). For example:

```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

**第四步：** 安装 MMEngine 和 MMCV, 并且我们建议使用 MIM 来完成安装

4: Install MMEngine and MMCV, and we recommend using MIM to complete the installation


```
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv>2.0.0rc4, <2.1.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**第五步：** 安装 MMDetection

5: Install MMDetection

```
mim install 'mmdet>3.0.0rc6, <3.2.0' -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**第六步：** 安装 ai4rs

6: Install ai4rs

```
git clone https://github.com/wokaikaixinxin/ai4rs.git
cd ai4rs
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```


## 数据准备  Data Preparation


请参考 [data_preparation.md](tools/data/README.md) 进行数据集准备

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data


```
ai4rs
├── data
│   ├── split_ss_dota
│   │   ├── trainval
│   │   ├── test
│   ├── split_ms_dota
│   │   ├── trainval
│   │   ├── test
│   ├── split_ss_dota1.5
│   │   ├── trainval
│   │   ├── test
│   ├── DIOR
│   │   ├── Annotations
│   │   │   ├─ Oriented Bounding Boxes
│   │   │   ├─ Horizontal Bounding Boxes
│   │   ├── ImageSets
│   │   │   ├─ Main
│   │   │   │  ├─ train.txt
│   │   │   │  ├─ val.txt
│   │   │   │  ├─ test.txt
│   │   ├── JPEGImages-test
│   │   ├── JPEGImages-trainval
│   ├── icdar2015
│   │   ├── ic15_textdet_train_img
│   │   ├── ic15_textdet_train_gt
│   │   ├── ic15_textdet_test_img
│   │   ├── ic15_textdet_test_gt
```

## 教程 Getting Started

请阅读[概述](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)对 Openmmlab 进行初步的了解。

Please see [Overview](https://mmrotate.readthedocs.io/en/1.x/overview.html) for the general introduction of Openmmlab.

为了帮助用户更进一步了解 Openmmlab，我们准备了用户指南和进阶指南，请阅读我们的[文档](https://mmrotate.readthedocs.io/zh_CN/1.x/)：

For detailed user guides and advanced guides, please refer to our [documentation](https://mmrotate.readthedocs.io/en/1.x/):


## 常见问题 FAQ

请参考 [FAQ](docs/en/notes/faq.md) 了解其他用户的常见问题。

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.


## 致谢 Acknowledgement

[OpenMMLab 官网](https://openmmlab.com)

[OpenMMLab 开放平台](https://platform.openmmlab.com)

[MMDetection](https://github.com/open-mmlab/mmdetection)

[MMRotate](https://github.com/open-mmlab/MMRotate)

## 引用 Citation

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 ai4rs

If you use this toolbox or benchmark in your research, please cite this project ai4rs

```bibtex

```



