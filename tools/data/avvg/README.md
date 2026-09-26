# Preparing AVVG Dataset

>[Geoground: A unified large vision-language model for remote sensing visual grounding](https://arxiv.org/abs/2411.11904)

>[GeoGround github](https://github.com/VisionXLab/GeoGround)


## Download AVVG dataset

The AVVG dataset can be downloaded from [official hugging face](https://huggingface.co/datasets/erenzhou/refGeo).



The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── refGeo
│   │   ├── images
│   │   │   │   ├── avvg
│   │   │   │   │   ├── xxx.JPG  # (623 JPG)
│   │   ├── metainfo
│   │   │   │   ├── avvg_detection_test.jsonl
│   │   │   │   ├── avvg_detection_train.jsonl
│   │   │   │   ├── avvg_test.jsonl
│   │   │   │   ├── avvg_train.jsonl
```



## Description

Remote sensing (RS) visual grounding aims to use natural language expression to locate specific objects (in the form of the bounding box or segmentation mask) in RS images, enhancing human interaction with intelligent RS interpretation systems. Early research in this area was primarily based on horizontal bounding boxes (HBBs), but as more diverse RS datasets have become available, tasks involving oriented bounding boxes (OBBs) and segmentation masks have emerged. In practical applications, different targets require different grounding types: HBB can localize an object's position, OBB provides its orientation, and mask depicts its shape. However, existing specialized methods are typically tailored to a single type of RS visual grounding task and are hard to generalize across tasks. In contrast, large vision-language models (VLMs) exhibit powerful multi-task learning capabilities but struggle to handle dense prediction tasks like segmentation. This paper proposes GeoGround, a novel framework that unifies support for HBB, OBB, and mask RS visual grounding tasks, allowing flexible output selection. Rather than customizing the architecture of VLM, our work aims to elegantly support pixel-level visual grounding output through the Text-Mask technique. We define prompt-assisted and geometry-guided learning to enhance consistency across different signals. To support model training, we present refGeo, a large-scale RS visual instruction-following dataset containing 161k image-text pairs. 

[Paper link](https://arxiv.org/abs/2411.11904)

<div align=center>
<img src="https://github.com/VisionXLab/GeoGround/blob/main/images/geoground/refgeo.jpg"  width="40%"/>
</div>


```bibtex
@article{zhou2024geoground,
  title={Geoground: A unified large vision-language model for remote sensing visual grounding},
  author={Zhou, Yue and Lan, Mengcheng and Li, Xiang and Feng, Litong and Ke, Yiping and Jiang, Xue and Li, Qingyun and Yang, Xue and Zhang, Wayne},
  journal={arXiv preprint arXiv:2411.11904},
  year={2024}
}
``` 