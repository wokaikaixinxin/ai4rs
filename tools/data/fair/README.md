# Preparing FAIR1M1.0 Dataset

>[FAIR1M: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery](https://www.gaofen-challenge.com/benchmark)



<!-- [DATASET] -->

## Download FAIR1M dataset

The FAIR1M dataset can be downloaded from [benchmark page](https://www.gaofen-challenge.com/benchmark) or [official baidu netdist](https://pan.baidu.com/share/init?surl=alWnbCbucLOQJJhi4WsZAw&pwd=u2xg) or [modelscope(魔塔)](https://www.modelscope.cn/datasets/wokaikaixinxin/FAIR1M1.0/files).

**How to use modelscope(魔塔) to download FAIR1M1.0**

1) Install `modelscope`

```shell
pip install modelscope
```

2) Download FAIR1M1.0

```shell
modelscope download --dataset 'wokaikaixinxin/FAIR1M1.0' --local_dir 'your_local_path'
```

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── FAIR1M1.0
│   │   ├── train
│   │   │   ├── part1
│   │   │   │   ├── images  (1732 tif)
│   │   │   │   ├── labelXml (1732 xml)
│   │   │   ├── part2
│   │   │   │   ├── images-1  (9487 tif)
│   │   │   │   ├── images-2  (5269 tif)
│   │   │   │   ├── labelXml  (14756 xml)
│   │   ├── test
│   │   │   ├── images  (8137 tif)
```

## split FAIR1M1.0 dataset

Please crop the original images into 1024×1024 patches with an overlap of 200 by run

```shell
python tools/data/fair/split/img_split.py --base-json \
  tools/data/fair/split/split_configs/fair1m1.0_ss_train.json

python tools/data/fair/split/img_split.py --base-json \
  tools/data/fair/split/split_configs/fair1m1.0_ss_test.json
```


Please update the `img_dirs` and `ann_dirs` in json.

The new data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── split_ss_fair1m1.0
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

Please change `data_root` in `configs/_base_/datasets/dota.py` to `data/split_ss_dota`.

## Classes of FAIR1M

The 37 classes including `other-ship`, `other-vehicle`, `other-airplane` are used. Details in the [FAIRDataset](../../../ai4rs/datasets/fair.py).

```
'classes':
(
    # ship 9
    'Passenger Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship','Motorboat',
    'Fishing Boat','Warship','Engineering Ship', 'other-ship', 'Tugboat',
    # vehicle 10
    'Small Car','Cargo Truck', 'Van', 'Trailer','other-vehicle','Dump Truck',
    'Bus', 'Tractor', 'Excavator', 'Truck Tractor',
    # airplane 11
    'Boeing737','Boeing747','Boeing777','Boeing787','other-airplane',
    'C919','A220','A321','A330','A350','ARJ21',
    # court 4
    'Tennis Court', 'Football Field', 'Basketball Court', 'Baseball Field',
    # road 3
    'Intersection', 'Bridge', 'Roundabout'
)
```

## Description

This benchmark provides a standard dataset for applying advanced deep learning technology to remote sensing. FAIR1M is a large-scale dataset for Fine-grAined object detection and recognItion in Remote sensing images. To meet the needs of practical applications, images in the FAIR1M dataset are collected from different sensors and platforms, with a spatial resolution ranging from 0.3m to 0.8m. There are more than 1 million instances and more than 40,000 images in this dataset. All objects in the FAIR1M dataset are annotated with respect to 5 categories and 37 sub-categories by oriented bounding boxes. Each image is of the size in the range from 1000 × 1000 to 10,000 × 10,000 pixels and contains objects exhibiting a wide variety of scales, orientations, and shapes.

[Paper link](https://arxiv.org/abs/2103.05569)

<div align=center>
<img src="https://pica.zhimg.com/v2-fa0551a68bb4931a519bb049f647a02c_1440w.jpg" />
</div>


```bibtex
@article{SUN2022116,
title = {FAIR1M: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {184},
pages = {116-130},
year = {2022},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2021.12.004}}
```