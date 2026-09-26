# Preparing DIOR-R-RSVG Dataset

>[A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing](https://arxiv.org/abs/2609.28230)


**DIOR-R-RSVG = DIOR-RSVG + Annotations_obb**

## Download DIOR-RSVG dataset first

[DIOR-RSVG github](https://github.com/zhanyang-nwpu/rsvg-pytorch)

[DIOR-RSVG GOOGLE Drive](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_)

## Download DIOR-R-RSVG dataset 'Annotations_obb.zip' second

The DIOR-R-RSVG dataset can be downloaded from [modelscope(魔塔)](https://modelscope.cn/datasets/wokaikaixinxin/dior_r_rsvg).

**How to use modelscope(魔塔) to download DIOR-R-RSVG**

1) Install `modelscope`

```shell
pip install modelscope
```

2) Download DIOR-R-RSVG

```shell
modelscope download --dataset wokaikaixinxin/dior_r_rsvg --local_dir 'your_local_path'
```

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── dior_rsvg
│   │   ├── Annotations_obb
│   │   │   │   ├── xxx.xml # (17402 xml) New.
│   │   ├── JPEGImages
│   │   │   │   ├── xxx.jpg # (17402 jpg) Same as DIOR-RSVG
│   │   ├── test.txt        # Same as DIOR-RSVG
│   │   ├── train.txt       # Same as DIOR-RSVG
│   │   ├── val.txt         # Same as DIOR-RSVG
```



## Description

Visual grounding in remote sensing images aims to locate objects described by referring expressions. Most existing methods predict horizontal bounding boxes, which are often inaccurate for objects with arbitrary orientations. To address this limitation, we introduce O2-VG, a family of models for oriented object visual grounding with three complementary designs. Specifically, O2-VG-Trans is a cross-modality transformer for oriented object visual grounding. It establishes a strong discriminative foundation for the model family. Building upon it, O2-VG-Uni predicts universal oriented proposals for possible foreground objects without specific text prompts. It also supports object retrieval through cached proposal embeddings. Using these universal oriented proposals as input prompts, O2-VG-VLM is an autoregressive vision-language model. It generates oriented box token blocks in parallel through multi-token prediction. In addition, we construct DIOR-R-RSVG, a dataset for oriented object visual grounding in remote sensing images. It provides image, expression, and oriented box triplets for training and evaluation. Together, the O2-VG family provides a flexible framework that spans discriminative transformers and generative vision-language models. It achieves superior performance across multiple benchmarks.

[Paper link](https://arxiv.org/abs/2609.28230)

<div align=center>
<img src='https://github.com/wokaikaixinxin/Eagle_o2_vg/blob/main/asset/dior_r_rsvg_history.png' width="50%"/>
</div>

<div align=center>
<img src='https://github.com/wokaikaixinxin/Eagle_o2_vg/blob/main/asset/dior_r_rsvg_obb.png' width="50%"/>
</div>

<div align=center>
<img src='https://github.com/wokaikaixinxin/Eagle_o2_vg/blob/main/asset/dior_r_rsvg_expression.png' width="50%"/>
</div>


```bibtex
@article{ding2026unified,
  title={A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing},
  author={Ding, Zeyu and Zhou, Yong and Zhao, Jiaqi and Du, Wen-Liang and
          Li, Xixi and Zhu, Hancheng and Yao, Rui and El Saddik, Abdulmotaleb},
  journal={arXiv preprint arXiv:2609.28230},
  year={2026}
}
``` 