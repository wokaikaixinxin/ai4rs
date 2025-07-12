# MMRotate-SAM

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/231659969-adf7dd4d-fcec-4677-9105-aa72b2ced00f.PNG"/>
</div>

The project folder holds codes related to ai4rs and SAM.

Script Descriptions:

1. `eval_zero-shot-oriented-detection_dota.py` implement Zero-shot Oriented Object Detection with SAM. It prompts SAM with predicted boxes from a horizontal object detector.
2. `demo_zero-shot-oriented-detection.py` inference single image for Zero-shot Oriented Object Detection with SAM.
3. `data_builder` holds configuration information and process of dataset, dataloader.

The project is refer to [playground](https://github.com/open-mmlab/playground).

## Installation

```shell
pip install git+https://gitee.com/mirrors/segment-anything.git
```

## Usage

**NOTE: the current path is `ai4rs`**

1. Inference MMRotate-SAM with a single image and obtain visualization result.

```shell
# download weights
mkdir ./work_dirs/mmrotate_sam
wget -P ./work_dirs/mmrotate_sam https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -P ./work_dirs/mmrotate_sam https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth
```

```
# demo
python ./projects/mmrotate-sam/demo_zero-shot-oriented-detection.py \
    ./data/split_ss_dota/test/images/P0006__1024__0___0.png \
    ./configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
    ./work_dirs/mmrotate_sam/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
    --sam-type "vit_b" --sam-weight ./work_dirs/mmrotate_sam/sam_vit_b_01ec64.pth --out-path output.png
```

If you want to save output masks from SAM,
```
python ./projects/mmrotate-sam/demo_zero-shot-oriented-detection.py \
    ./data/split_ss_dota/test/images/P0006__1024__0___0.png \
    ./configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
    ./work_dirs/mmrotate_sam/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
    --sam-type "vit_b" --sam-weight ./work_dirs/mmrotate_sam/sam_vit_b_01ec64.pth --out-path output.png \
    --save-masks True --masks-path './mmrotate_sam_output_masks/'
```

<div align=center>
<img src="https://user-images.githubusercontent.com/79644233/231568599-58694ec9-a3b1-44a4-833f-74cfb4d4ca45.png"/>
</div>

2. Evaluate the quantitative evaluation metric on DOTA data set.

```shell
python ./projects/mmrotate-sam/eval_zero-shot-oriented-detection_dota.py \
    ./configs/rotated_fcos/rotated-fcos-hbox-le90_r50_fpn_1x_dota.py \
    ./work_dirs/mmrotate_sam/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth \
    --sam-type "vit_b" --sam-weight ./work_dirs/mmrotate_sam/sam_vit_b_01ec64.pth
```