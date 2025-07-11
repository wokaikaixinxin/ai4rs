# Preparing SARDet 100K Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{karatzas2015icdar,
@inproceedings{li2024sardet100k,
	title={SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection}, 
	author={Yuxuan Li and Xiang Li and Weijie Li and Qibin Hou and Li Liu and Ming-Ming Cheng and Jian Yang},
	year={2024},
	booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
}
}
```

## Download SARDet 100K dataset

The ICDAR2015 dataset can be downloaded from   
[ModelScope](https://www.modelscope.cn/datasets/wokaikaixinxin/SARDet_100K/files) or  
[baidu wangpan](https://pan.baidu.com/s/1dIFOm4V2pM_AjhmkD1-Usw?pwd=SARD) or  
[OneDrive](https://www.kaggle.com/datasets/greatbird/sardet-100k)

The data structure is as follows:

```none
ai4rs
├── data
│   ├── SARDet_100K
│   │   ├── Annotations
│   │   │   ├── test.json
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   ├── JPEGImages
│   │   │   ├── test
│   │   │   │   ├── 0000018.png
│   │   │   │   ├── xxxxxxx.png
│   │   │   ├── train
│   │   │   │   ├── xxxxxxx.png
│   │   │   │   ├── xxxxxxx.png
│   │   │   ├── val
│   │   │   │   ├── xxxxxxx.png
│   │   │   │   ├── xxxxxxx.png
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/sardet_100k.py` to `data/SARDet_100K/`.
