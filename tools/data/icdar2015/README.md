# Preparing ICDAR2015 Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{karatzas2015icdar,
  title={ICDAR 2015 competition on robust reading},
  author={Karatzas, Dimosthenis and Gomez-Bigorda, Lluis and Nicolaou, Anguelos and Ghosh, Suman and Bagdanov, Andrew and Iwamura, Masakazu and Matas, Jiri and Neumann, Lukas and Chandrasekhar, Vijay Ramaseshan and Lu, Shijian and others},
  booktitle={2015 13th international conference on document analysis and recognition (ICDAR)},
  pages={1156--1160},
  year={2015},
  organization={IEEE}
}
```

## Download ICDAR2015 dataset

The ICDAR2015 dataset can be downloaded from [official link](https://rrc.cvc.uab.es/?ch=4&com=introduction) or [ModelScope](https://www.modelscope.cn/datasets/wokaikaixinxin/icdar2015/)

The data structure is as follows:

```none
ai4rs
├── data
│   ├── icdar2015
│   │   ├── ic15_textdet_train_img
│   │   ├── ic15_textdet_train_gt
│   │   ├── ic15_textdet_test_img
│   │   ├── ic15_textdet_test_gt
```

## Change base config

Please change `data_root` in `configs/_base_/datasets/icdar2015.py` to `data/idcar2015/`.
