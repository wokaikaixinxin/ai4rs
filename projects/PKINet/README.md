# Poly Kernel Inception Network for Remote Sensing Detection

> [Poly Kernel Inception Network for Remote Sensing Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://mmbiz.qpic.cn/sz_mmbiz_png/GKEYUpEACeAGwrGtk0DYZIEPj9icuwYicE2icmVBnGPq6xyNPLsL7elVP7DtNg0u606GyGS1kS6fow0D0t01AvyAw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1" width="800"/>
</div>

Object detection in remote sensing images (RSIs) often suffers from several increasing challenges including the large variation in object scales and the diverse-ranging context. Prior methods tried to address these challenges by expanding the spatial receptive field of the backbone either through large-kernel convolution or dilated convolution. However the former typically introduces considerable background noise while the latter risks generating overly sparse feature representations. In this paper we introduce the Poly Kernel Inception Network (PKINet) to handle the above challenges. PKINet employs multi-scale convolution kernels without dilation to extract object features of varying scales and capture local context. In addition a Context Anchor Attention (CAA) module is introduced in parallel to capture long-range contextual information. These two components work jointly to advance the performance of PKINet on four challenging remote sensing object detection benchmarks namely DOTA-v1.0 DOTA-v1.5 HRSC2016 and DIOR-R.

## Results and models

NOTE: We **donnot** reimplement the experiment. The results and logs come from [official github](https://github.com/PKINet/PKINet).

DOTA1.0

|         Backbone         |  AP50  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PKINet-T (1024,1024,200) | 77.87 | le90  |   30e    |   -   |      -      |  -  | 16=4gpu*4img/gpu | [pkinet_t_fpn_o-rcnn_dota1.0_ss_le90](./configs/pkinet_t_fpn_o-rcnn_dota1.0_ss_le90.py) | [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/PKINet/pkinet_t_o-rcnn_dotav1-ss.pth) \| log |
| PKINet-S (1024,1024,200) | 78.39 | le90  |   30e    |   -   |      -      |  -  |  8=4gpu*2img/gpu  |             [pkinet_s_fpn_o-rcnn_dota1.0_ss_le90](./configs/pkinet_s_fpn_o-rcnn_dota1.0_ss_le90.py)              |         [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/PKINet/pkinet_s_o-rcnn_dotav1-ss.pth) \| [log](https://github.com/user-attachments/files/16598304/20231105_214905.log)         |

NOTE: I am **not sure** if the batch size of PKINet-T is 16(4gpu*4img/gpu). Please try it yourself.

NOTE: We **donnot** reimplement the experiment. The results and logs come from [official github](https://github.com/PKINet/PKINet).

DOTA1.5

|         Backbone         |  AP50  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PKINet-S (1024,1024,200) | 71.47 | le90  |   30e    |   -   |     -      |  -  |   8=4gpu*2img/gpu   |                 [pkinet_s_fpn_o-rcnn_dota1.5_ss_le90](./configs/pkinet_s_fpn_o-rcnn_dota1.5_ss_le90.py)                  |                   [model](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/PKINet/pkinet_s_o-rcnn_dotav15-ss.pth) \| log   



## Citation

```
@InProceedings{Cai_2024_CVPR,
    author    = {Cai, Xinhao and Lai, Qiuxia and Wang, Yuwei and Wang, Wenguan and Sun, Zeren and Yao, Yazhou},
    title     = {Poly Kernel Inception Network for Remote Sensing Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27706-27716}
}
```
