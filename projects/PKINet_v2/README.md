# PKINet-v2: Towards Powerful and Efficient Poly-Kernel Remote Sensing Object Detection

> [PKINet-v2: Towards Powerful and Efficient Poly-Kernel Remote Sensing Object Detection](https://arxiv.org/abs/2603.16341)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://github.com/PKINet/PKINet/blob/main/assets/PKINet-v2_pipeline.png" width="800"/>
</div>

Object detection in remote sensing images (RSIs) is challenged by the coexistence of geometric and spatial complexity: targets may appear with diverse aspect ratios, while spanning a wide range of object sizes under varied contexts. Existing RSI backbones address the two challenges separately, either by adopting anisotropic strip kernels to model slender targets or by using isotropic large kernels to capture broader context. However, such isolated treatments lead to complementary drawbacks: the strip-only design can disrupt spatial coherence for regular-shaped objects and weaken tiny details, whereas isotropic large kernels often introduce severe background noise and geometric mismatch for slender structures. In this paper, we extend PKINet, and present a powerful and efficient backbone that jointly handles both challenges within a unified paradigm named Poly Kernel Inception Network v2 (PKINet-v2). PKINet-v2 synergizes anisotropic axial-strip convolutions with isotropic square kernels and builds a multi-scope receptive field, preserving fine-grained local textures while progressively aggregating long-range context across scales. To enable efficient deployment, we further introduce a Heterogeneous Kernel Re-parameterization (HKR) Strategy that fuses all heterogeneous branches into a single depth-wise convolution for inference, eliminating fragmented kernel launches without accuracy loss

## Results and models

NOTE: We **donnot** reimplement the experiment. The results and logs come from [official github](https://github.com/PKINet/PKINet).

DOTA1.0

| Model | Detector | mAP | Angle | Config | Dev-Config | Weights |
|:--|:--|:--:|:--:|:--|:--|:--|
| PKINet-v2-T | Oriented R-CNN | 79.36 | le90 | [config](./configs/pkinet-v2-t_fpn_o-rcnn_3x_dotav1-ss_le90.py) | - | [model](https://1drv.ms/u/c/9ce9a57f1a400a74/IQABDi6m2KwFSKmu9H2Nvye0ASdS5RcIhcEIYyelTtS6au8?e=6cvw5Z) |


NOTE: I am **not sure** if the batch size of PKINet-v2-T is 16(8gpu*2img/gpu). Please try it yourself.

NOTE: We **donnot** reimplement the experiment. The results and logs come from [official github](https://github.com/PKINet/PKINet).


## Citation

```
@article{cai2026pkinetv2,
  title={PKINet-v2: Towards Powerful and Efficient Poly-Kernel Remote Sensing Object Detection},
  author={Cai, Xinhao and Liulei Li and Gensheng Pei and Zeren Sun and Yazhou Yao and Wenguan Wang},
  journal={arXiv preprint arXiv:2603.16341},
  year={2026}
}
```
