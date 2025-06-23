<div align="center">
  <img src="resources/ai4rs-logo.png" width="800"/>
</div>



<div align="center">

[📘使用文档](https://mmrotate.readthedocs.io/zh_CN/1.x/) &#124;
[🛠️安装教程](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) &#124;
[👀模型库](https://mmrotate.readthedocs.io/zh_CN/1.x/model_zoo.html) &#124;
[🆕更新日志](https://mmrotate.readthedocs.io/en/1.x/notes/changelog.html)

</div>


<p align="center">
 简体中文 | <a href="/README_en.md">English</a>
</p>


## 介绍

AI for Remote Sensing 是一款基于 PyTorch 的人工智能与遥感结合的开源工具箱。


人工智能发展很快，相关工作很多。希望在MMLab基础上，特别是mmdetection、mmrotate的基础上集成遥感相关的工作。虽然MMLab的很多仓库已经停止更新，但薪尽火传。


<details open>
<summary><b>主要特性</b></summary>

- **支持多种角度表示法**

  MMRotate 提供了三种主流的角度表示法以满足不同论文的配置。

- **模块化设计**

  MMRotate 将旋转框检测任务解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的旋转框检测算法模型。

- **强大的基准模型与SOTA**

  MMRotate 提供了旋转框检测任务中最先进的算法和强大的基准模型.

</details>

## 最新进展

### 亮点



## 安装

请参考[快速入门文档](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)进行安装。

## 教程

请阅读[概述](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html)对 MMDetection 进行初步的了解。

为了帮助用户更进一步了解 MMDetection，我们准备了用户指南和进阶指南，请阅读我们的[文档](https://mmrotate.readthedocs.io/zh_CN/1.x/)：

- 用户指南
  - [训练 & 测试](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/index.html#train-test)
    - [学习配置文件](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/config.html)
    - [使用已有模型在标准数据集上进行推理](https://mmrotate.readthedocs.io/en/1.x/user_guides/inference.html)
    - [数据集准备](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/dataset_prepare.html)
    - [测试现有模型](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/train_test.html#test)
    - [在标准数据集上训练预定义的模型](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/train_test.html#train)
    - [提交测试结果](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/test_results_submission.html)
  - [实用工具](https://mmrotate.readthedocs.io/zh_CN/1.x/user_guides/index.html#useful-tools)
- 进阶指南
  - [基础概念](https://mmrotate.readthedocs.io/zh_CN/1.x/advanced_guides/index.html#basic-concepts)
  - [组件定制](https://mmrotate.readthedocs.io/zh_CN/1.x/advanced_guides/index.html#component-customization)
  - [How to](https://mmrotate.readthedocs.io/zh_CN/1.x/advanced_guides/index.html#how-to)

我们提供了旋转检测的 colab 教程 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](demo/MMRotate_Tutorial.ipynb)。

若需要将0.x版本的代码迁移至新版，请参考[迁移文档](https://mmrotate.readthedocs.io/zh_CN/1.x/migration.html)。

## 模型库

各个模型的结果和设置都可以在对应的 config（配置）目录下的 *README.md* 中查看。
整体的概况也可也在 [模型库](docs/zh_cn/model_zoo.md) 页面中查看。

<details open>
<summary><b>Oriented Object Detection</b></summary>

|     |     |     |     |
| :---: | :---: | :---: | :---: |
| [Rotated RetinaNet-OBB/HBB<br>(ICCV'2017)](configs/rotated_retinanet/README.md) | [Rotated FasterRCNN-OBB<br>(TPAMI'2017)](configs/rotated_faster_rcnn/README.md) | [Rotated RepPoints-OBB<br>(ICCV'2019)](configs/rotated_reppoints/README.md) | [Rotated FCOS<br>(ICCV'2019)](configs/rotated_fcos/README.md) |
| [RoI Transformer<br>(CVPR'2019)](configs/roi_trans/README.md) | [Gliding Vertex<br>(TPAMI'2020)](configs/gliding_vertex/README.md) | [Rotated ATSS-OBB<br>(CVPR'2020)](configs/rotated_atss/README.md) | [CSL<br>(ECCV'2020)](configs/csl/README.md) |
| [R<sup>3</sup>Det<br>(AAAI'2021)](configs/r3det/README.md) | [S<sup>2</sup>A-Net<br>(TGRS'2021)](configs/s2anet/README.md) | [ReDet<br>(CVPR'2021)](configs/redet/README.md) | [Beyond Bounding-Box<br>(CVPR'2021)](configs/cfa/README.md) |
| [Oriented R-CNN<br>(ICCV'2021)](configs/oriented_rcnn/README.md) | [GWD<br>(ICML'2021)](configs/gwd/README.md) | [KLD<br>(NeurIPS'2021)](configs/kld/README.md) | [SASM<br>(AAAI'2022)](configs/sasm_reppoints/README.md) |
| [Oriented RepPoints<br>(CVPR'2022)](configs/oriented_reppoints/README.md) | [KFIoU<br>(ICLR'2023)](configs/kfiou/README.md) | [H2RBox<br>(ICLR'2023)](configs/h2rbox/README.md) | [PSC<br>(CVPR'2023)](configs/psc/README.md) |
| [RTMDet<br>(arXiv)](configs/rotated_rtmdet/README.md) | [H2RBox-v2<br>(arXiv)](configs/h2rbox_v2/README.md)



</details>

## 数据准备

请参考 [data_preparation.md](tools/data/README.md) 进行数据集准备。

## 常见问题

请参考 [FAQ](docs/en/notes/faq.md) 了解其他用户的常见问题。

## 参与贡献

我们非常欢迎用户对于 MMRotate 做出的任何贡献，可以参考 [CONTRIBUTION.md](.github/CONTRIBUTING.md) 文件了解更多细节。

## 致谢

[OpenMMLab 官网](https://openmmlab.com)

[OpenMMLab 开放平台](https://platform.openmmlab.com)

[MMDetection](https://github.com/open-mmlab/mmdetection)

[MMRotate](https://github.com/open-mmlab/MMRotate)

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 ai4rs。

```bibtex

```



