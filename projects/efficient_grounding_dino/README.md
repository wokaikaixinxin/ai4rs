# Efficient Grounding DINO: Efficient Cross-Modality Fusion and Efficient Label Assignment for Visual Grounding in Remote Sensing


[TGRS 2025 Efficient Grounding Dino](https://ieeexplore.ieee.org/abstract/document/10857369)

[Official Github Repo](https://github.com/Gao-Kun-Lab/Efficient-Grounding-DINO)

## Abstract

Visual grounding for remote sensing (RSVG) aims to detect objects in remote sensing scenes based on textual descriptions. While existing methods perform well on RSVG datasets, they are limited to single-object predictions, making them unsuitable for multi-object candidate category datasets. Open-set methods can be applied to both RSVG and candidate datasets, but their use in remote sensing remains rare. To bridge this gap, we introduce the open-set approach to RSVG and propose Efficient Grounding DINO, using Grounding DINO as a baseline. Open-set methods rely on two key modules: cross-modality fusion and label assignment. Existing cross-modality fusion methods simultaneously update text and multi-scale visual features, which hampers the model’s ability to generalize under different texts and increases learning complexity. Existing methods predict a single object, allowing direct use as a positive example for loss calculation, while open-set methods for multi-objects require one-to-one matching to assign positive and negative samples. However, background interference in the RSVG datasets causes frequent misassignments, slowing model convergence. We address these issues with two innovations: the multi-scale image-to-text fusion module (MSITFM), which updates text features using self-attention to maintain independence from visual features and employs scale-specific cross-attention for multi-scale visual feature fusion to reduce learning complexity, achieving a 3% parameter and 21.6% GFLOPs reduction. Text confidence matching (TCM) incorporates IoU-based confidence into label assignment to reduce mismatches and enhance model performance


<div align=center>
<img src='https://github.com/Gao-Kun-Lab/Efficient-Grounding-DINO/blob/main/assets/readme/image.png' width="80%"/>
</div>


## Preparation -- BERT Model

Download BERT

```
# you can set hugging face mirror
export HF_ENDPOINT=https://hf-mirror.com
# download
huggingface-cli download google-bert/bert-base-uncased --local-dir /root/bert-base-uncased
```
Modified BERT Path in [grounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG.py](./configs/grounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG.py#L19)

## Installation

```shell
pip install fairscale -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Main Results

**DIOR-RSVG**

| Method | Backbone | Pr@0.5 | Config | Download |
| :----: | :------: | :--: | :-----: | :------: |
|efficient grounding dino| R50 (640,640) | 80.36 |    [grounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG](./configs/grounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG.py)      |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/efficient_grounding_dino%2Fgrounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG%2Fepoch_12.pth) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/efficient_grounding_dino/grounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/efficient_grounding_dino%2Fgrounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG%2F20260729_234047.log) |

| Method | Backbone | Pr@0.5 | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cumIoU |
| :----: | :------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|efficient grounding dino| R50 (640,640) | 80.36 | 77.32 | 71.57 | 60.96 | 40.31 | 70.94 | 80.96 |

**Note**: The official repository does not provide weights. This is our reimplementation.

**Note**: The official repository does not provide weights. This is our reimplementation.

**Note**: The batch size is **16** in the official code, while it is **8 (2gpu * 4img/gpu = 8)** in our implementation.


train

```Shell
bash tools/dist_train.sh projects/efficient_grounding_dino/configs/grounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG.py 2
```

test

```Shell
bash tools/dist_test.sh projects/efficient_grounding_dino/configs/grounding_dino_fusion_decouple_IoU_match_r50_scratch_2xb4_1x_RSVG.py work_dirs/grounding_dino_fusion_decouple_IoU_match_r50_scratch_8xb2_1x_RSVG/epoch_12.pth 2
```


## Citation

```bibtex
@ARTICLE{10857369,
  author={Hu, Zibo and Gao, Kun and Zhang, Xiaodian and Yang, Zhijia and Cai, Mingfeng and Zhu, Zhenyu and Li, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Efficient Grounding DINO: Efficient Cross-Modality Fusion and Efficient Label Assignment for Visual Grounding in Remote Sensing}, 
  year={2025},
  volume={63},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2025.3536015}}
``` 