# Preparing VRSBench Dataset

>[VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding](https://arxiv.org/abs/2406.12384)

>[VRSBench github](https://github.com/lx709/VRSBench)


## Download VRSBench dataset

The VRSBench dataset can be downloaded from [official hugging face](https://huggingface.co/datasets/xiang709/VRSBench) .

The data structure is as follows:

```none
ai4rs
├── mmrotate
├── tools
├── configs
├── data
│   ├── VRSBench
│   │   ├── Annotations_train
│   │   │   │   ├── xxx.json # (20264 json)
│   │   ├── Annotations_val
│   │   │   │   ├── xxx.json # (9350 json)
│   │   ├── Images_train
│   │   │   │   ├── xxx.png # (20264 png)
│   │   ├── Images_val
│   │   │   │   ├── xxx.png # (9350 png)
│   │   ├── VRSBench_EVAL_referring.json
│   │   ├── VRSBench_train.json
```



## Description

We introduce a new benchmark designed to advance the development of general-purpose, large-scale vision-language models for remote sensing images. Although several vision-language datasets in remote sensing have been proposed to pursue this goal, existing datasets are typically tailored to single tasks, lack detailed object information, or suffer from inadequate quality control. Exploring these improvement opportunities, we present a Versatile vision-language Benchmark for Remote Sensing image understanding, termed VRSBench. This benchmark comprises 29,614 images, with 29,614 human-verified detailed captions, 52,472 object references, and 123,221 question-answer pairs. It facilitates the training and evaluation of vision-language models across a broad spectrum of remote sensing image understanding tasks. We further evaluated state-of-the-art models on this benchmark for three vision-language tasks: image captioning, visual grounding, and visual question answering. Our work aims to significantly contribute to the development of advanced vision-language models in the field of remote sensing.

[Paper link](https://arxiv.org/abs/2406.12384)

[Nips link](https://neurips.cc/virtual/2024/poster/97530)

<div align=center>
<img src="https://github.com/lx709/VRSBench/blob/main/fig_example.png" />
</div>


```bibtex
@article{li2024vrsbench,
  title={Vrsbench: A versatile vision-language benchmark dataset for remote sensing image understanding},
  author={Li, Xiang and Ding, Jian and Elhoseiny, Mohamed},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={3229--3242},
  year={2024}
}
``` 