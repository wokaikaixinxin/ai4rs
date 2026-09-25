# A Unified Framework and Dataset for Oriented Object Visual Grounding in Remote Sensing -- O2-VG-Uni


[Arxiv](https://arxiv.org/abs/2609.28230)




## Abstract

Visual grounding in remote sensing images aims to locate objects described by referring expressions. Most existing methods predict horizontal bounding boxes, which are often inaccurate for objects with arbitrary orientations. To address this limitation, we introduce O2-VG, a family of models for oriented object visual grounding with three complementary designs. Specifically, O2-VG-Trans is a cross-modality transformer for oriented object visual grounding. It establishes a strong discriminative foundation for the model family. Building upon it, O2-VG-Uni predicts universal oriented proposals for possible foreground objects without specific text prompts. It also supports object retrieval through cached proposal embeddings. Using these universal oriented proposals as input prompts, O2-VG-VLM is an autoregressive vision-language model. It generates oriented box token blocks in parallel through multi-token prediction. In addition, we construct DIOR-R-RSVG, a dataset for oriented object visual grounding in remote sensing images. It provides image, expression, and oriented box triplets for training and evaluation. Together, the O2-VG family provides a flexible framework that spans discriminative transformers and generative vision-language models. It achieves superior performance across multiple benchmarks.

## Main Results

**DIOR-R-RSVG**

| Method | Backbone | stage 1 config | Download |
| :----: | :------: | :-----: | :------: |
|O2-VG-Uni| R50vd (800,800) | [ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_dior_r_rsvg](./configs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_dior_r_rsvg.py) |  [24th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_dior_r_rsvg%2Fepoch_24.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_dior_r_rsvg%2F20260810_054338%2F20260810_054338.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_ov_rtdetr/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_dior_r_rsvg) |


| Method | Backbone | stage 2 config | Download |
| :----: | :------: | :-----: | :------: |
|O2-VG-Uni| R50vd (800,800) | [ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_dior_r_rsvg](./configs/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_dior_r_rsvg.py) |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_dior_r_rsvg%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_ov_rtdetr/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_dior_r_rsvg/20260810_225921/20260810_225921.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_ov_rtdetr/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_dior_r_rsvg) |


|   Method  |     Backbone    | AR@100 | AR@300 | AR@1000 |
| :----: | :------: | :-----: | :------: | :------: |
|O2-VG-Trans| R50vd (800,800) | 0.8623  | 0.8853  | 0.8853  |


```Shell
# train stage 1
bash tools/dist_train.sh projects/rotated_ov_rtdetr/configs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_dior_r_rsvg.py 2
# train stage 2
bash tools/dist_train.sh projects/rotated_ov_rtdetr/configs/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_dior_r_rsvg.py 2
```

**VRSBench**

| Method | Backbone | stage 1 config | Download |
| :----: | :------: | :-----: | :------: |
|O2-VG-Uni| R50vd (512,512) | [ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_vrsbench](./configs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_vrsbench.py)  |  [24th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_vrsbench%2Fepoch_24.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_vrsbench%2F20260809_015331%2F20260809_015331.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_ov_rtdetr/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_vrsbench) |


| Method | Backbone | stage 2 config | Download |
| :----: | :------: | :-----: | :------: |
|O2-VG-Uni| R50vd (512,512) | [ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_vrsbench](./configs/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_vrsbench.py)  |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_vrsbench%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_ov_rtdetr/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_vrsbench/20260809_205922/20260809_205922.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_ov_rtdetr/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs8_12e_vrsbench) |


|   Method  |     Backbone    | AR@100 | AR@300 | AR@1000 |
| :----: | :------: | :-----: | :------: | :------: |
|O2-VG-Trans| R50vd (512,512) | 0.8679 | 0.8789 | 0.8789  |


```Shell
# train stage 1
bash tools/dist_train.sh projects/rotated_ov_rtdetr/configs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_vrsbench.py 2
# train stage 2
bash tools/dist_train.sh projects/rotated_ov_rtdetr/configs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs8_24e_vrsbench.py 2
```


**AVVG**

| Method | Backbone | stage 1 config | Download |
| :----: | :------: | :-----: | :------: |
|O2-VG-Uni| R50vd (1024, 576) | [ov_o2_rtdetr_r50vd_rmclip_vitb32_bs4_72e_avvg](./configs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs4_72e_avvg.py)  |  [72th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_r50vd_rmclip_vitb32_bs4_72e_avvg%2Fepoch_72.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_ov_rtdetr/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs4_72e_avvg/20260809_022326/20260809_022326.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_ov_rtdetr/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs4_72e_avvg) |


| Method | Backbone | stage 2 config | Download |
| :----: | :------: | :-----: | :------: |
|O2-VG-Uni| R50vd (1024, 576) | [ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs4_12e_avvg](./configs/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs4_12e_avvg.py)  |  [12th_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/file/view/master/rotated_ov_rtdetr%2Fov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs4_12e_avvg%2Fepoch_12.pth) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_ov_rtdetr/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs4_12e_avvg/20260809_221502/20260809_221502.log) \| [all_ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/tree/master/rotated_ov_rtdetr/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs4_12e_avvg) |


|   Method  |     Backbone    | AR@100 | AR@300 | AR@1000 |
| :----: | :------: | :-----: | :------: | :------: |
|O2-VG-Trans| R50vd (1024, 576) | 0.9728  | 0.9853  |  0.9853 |


```Shell
# train stage 1
bash tools/dist_train.sh projects/rotated_ov_rtdetr/configs/ov_o2_rtdetr_r50vd_rmclip_vitb32_bs4_72e_avvg.py 2
# train stage 2
bash tools/dist_train.sh projects/rotated_ov_rtdetr/configs/ov_o2_rtdetr_uni_r50vd_rmclip_vitb32_bs4_12e_avvg.py 2
```


## Visualization Results Demo




## Bibtex

