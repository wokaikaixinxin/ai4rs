# Preparing KFGOD Dataset

>[KFGOD: A Fine-Grained Object Detection Dataset in KOMPSAT Satellite Imagery](https://www.mdpi.com/2072-4292/17/22/3774)


## Download KFGOD dataset

The KFGOD dataset can be downloaded from [dataon](https://doi.org/10.22711/idr/1101) or [huggingface](https://huggingface.co/datasets/ohhan777/kfgod) or [modelscope(魔塔)](https://modelscope.cn/datasets/wokaikaixinxin/KFGOD).



The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── KFGOD
│   │   ├── train
│   │   │   │   ├── annfile # (3070 txt)
│   │   │   │   ├── images  # (3073 png)
│   │   ├── val
│   │   │   │   ├── annfile # (469 txt)
│   │   │   │   ├── images  # (470 png)
```

In train set, images `OBJ00209_PS3_K3_AIDATA0601`, `OBJ01530_PS3_K3A_AIDATA0638`, and `OBJ01735_PS3_K3_AIDATA0009` have no annotation files.  
In val set, image `OBJ01677_PS3_K3A_AIDATA0145` has no annotation files.

## Classes of KFGOD

The 33 classes.

```
'classes':
(
    'aquaculture_farm', 'barge', 'bridge', 'bus', 'container', 'container_group',
    'container_ship', 'crane', 'dam', 'drill_ship', 'ferry', 'fighter_aircraft',
    'fishing_boat', 'helicopter', 'helipad',
    'large_civilian_aircraft', 'large_military_aircraft', 'marine_research_station',
    'motorboat', 'oil_tanker', 'roundabout', 'sailboat', 'small_civilian_aircraft', 'small_vehicle',
    'sports_field', 'stadium', 'storage_tank', 'swimming_pool', 'train', 'truck',
    'tugboat', 'warship', 'wind_turbine'
)
```



## Description

KFGOD provides approximately 880K object instances across 33 fine-grained classes from homogeneous KOMPSAT-3/3A imagery (0.55–0.7 m resolution), with dual OBB+HBB annotations.

The dataset’s unique sensor homogeneity (KOMPSAT-3/3A only) provides a wellcontrolled, sensor-consistent benchmark that minimizes sensor-induced domain gaps and enables a fair comparison of the detection algorithms.

[Paper link](https://doi.org/10.22711/idr/1101)

<div align=center>
<img src="https://mdpi-res.com/remotesensing/remotesensing-17-03774/article_deploy/html/images/remotesensing-17-03774-g002-550.jpg" />
</div>


```bibtex

@Article{rs17223774,
AUTHOR = {Lee, Dong Ho and Hong, Ji Hun and Seo, Hyun Woo and Oh, Han},
TITLE = {KFGOD: A Fine-Grained Object Detection Dataset in KOMPSAT Satellite Imagery},
JOURNAL = {Remote Sensing},
VOLUME = {17},
YEAR = {2025},
NUMBER = {22},
ARTICLE-NUMBER = {3774},
URL = {https://www.mdpi.com/2072-4292/17/22/3774},
ISSN = {2072-4292},
DOI = {10.3390/rs17223774}
}
``` 