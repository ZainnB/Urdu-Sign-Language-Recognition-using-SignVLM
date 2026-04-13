# SignVLM: A Pre-trained Large Vision Model for Sign Language Recognition

This is the official implementation of the paper "SignVLM: A Pre-trained Large Vision Model for Sign Language Recognition"



## Introduction

The overall architecture of the framework includes a trainable Transformer decoder, trainable local temporal modules and pre-trained models, fixed image backbone
(CLIP is used for this work).


## Installation

We tested the released code with the following conda environment

```
conda create -n pt1.9.0cu11.1_official -c pytorch -c conda-forge pytorch=1.9.0=py3.9_cuda11.1_cudnn8.0.5_0 cudatoolkit torchvision av
```

## Data Preparation

We expect that `--train_list_path` and `--val_list_path` command line arguments to be a data list file of the following format
```
<path_1> <label_1>
<path_2> <label_2>
...
<path_n> <label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.
`--num_classes` should also be specified in the command line argument.

Additionally, `<path_i>` might be a relative path when `--data_root` is specified, and the actual path will be
relative to the path passed as `--data_root`.

## Datasets Download

The datasets used in this paper are publicly available and can be downloaded as follows:
- [KArSL](https://hamzah-luqman.github.io/KArSL/)
- [WLASL](https://dxli94.github.io/WLASL/)
- [LSA64](https://facundoq.github.io/datasets/lsa64/)
- [AUTSL](https://cvml.ankara.edu.tr/datasets/)
  

## Backbone Preparation

CLIP weights need to be downloaded from [CLIP official repo](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L30)
and passed to the `--backbone_path` command line argument.

## Script Usage

Training and evaliation scripts are provided in the scripts folder.
Scripts should be ready to run once the environment is setup and 
`--backbone_path`, `--train_list_path` and `--val_list_path` are replaced with your own paths.

For other command line arguments please see the help message for usage.

## Acknowledgements

We would like to knowledge [PySlowFast](https://github.com/facebookresearch/SlowFast) and [EfficeintVideoRecognition](https://github.com/OpenGVLab/efficient-video-recognition) for sharing their codes. Thanks for their awesome work!
