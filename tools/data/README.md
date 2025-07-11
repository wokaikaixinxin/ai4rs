# Data Preparation for Rotation Detection

It is recommended to symlink the dataset root to `$ai4rs/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

Datasets supported in ai4rs:

- [DOTA Dataset](dota/README.md) \[ [Homepage](https://captain-whu.github.io/DOTA/) \]
- [DIOR Dataset](dior/README.md) \[ [Homepage](https://gcheng-nwpu.github.io/#Datasets) \]
- [SSDD Dataset](ssdd/README.md)
- [HRSC Dataset](hrsc/README.md)
- [HRSID Dataset](hrsid/README.md)
- [SRSDD Dataset](srsdd/README.md)
- [RSDD Dataset](rsdd/README.md)
- [ICDAR2015 Dataset](icdar2015/README.md)
- [SARDet 100K Dataset](./sardet_100k/README.md)


```
ai4rs
├── data
│   ├── split_ss_dota
│   │   ├── trainval
│   │   ├── test
│   ├── split_ms_dota
│   │   ├── trainval
│   │   ├── test
│   ├── split_ss_dota1.5
│   │   ├── trainval
│   │   ├── test
│   ├── DIOR
│   │   ├── Annotations
│   │   │   ├─ Oriented Bounding Boxes
│   │   │   ├─ Horizontal Bounding Boxes
│   │   ├── ImageSets
│   │   │   ├─ Main
│   │   │   │  ├─ train.txt
│   │   │   │  ├─ val.txt
│   │   │   │  ├─ test.txt
│   │   ├── JPEGImages-test
│   │   ├── JPEGImages-trainval
│   ├── icdar2015
│   │   ├── ic15_textdet_train_img
│   │   ├── ic15_textdet_train_gt
│   │   ├── ic15_textdet_test_img
│   │   ├── ic15_textdet_test_gt
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
