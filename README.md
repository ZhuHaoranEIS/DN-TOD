# Robust Tiny Object Detection in Aerial Images amidst Label Noise
This is the official implementation of the paper "Robust Tiny Object Detection in Aerial Images amidst Label Noise". [arxiv](https://arxiv.org/abs/2401.08056)

## Introduction
DN-TOD is an effective denoising algorithm that can be integrated into either one-stage or two-stage algorithms to enhance the network's robustness against noise.

**Abstract**: Precise detection of tiny objects in remote sensing imagery remains a significant challenge due to their limited visual information and frequent occurrence within scenes. This challenge is further exacerbated by the practical burden and inherent errors associated with manual annotation: annotating tiny objects is laborious and prone to errors (i.e., label noise). Training detectors for such objects using noisy labels often leads to suboptimal performance, with networks tending to overfit on noisy labels. In this study, we address the intricate issue of tiny object detection under noisy label supervision. We systematically investigate the impact of various types of noise on network training, revealing the vulnerability of object detectors to class shifts and inaccurate bounding boxes for tiny objects. To mitigate these challenges, we propose a DeNoising Tiny Object Detector (DN-TOD), which incorporates a Class-aware Label Correction (CLC) scheme to address class shifts and a Trend-guided Learning Strategy (TLS) to handle bounding box noise. CLC mitigates inaccurate class supervision by identifying and filtering out class-shifted positive samples, while TLS reduces noisy box-induced erroneous supervision through sample reweighting and bounding box regeneration. Additionally, Our method can be seamlessly integrated into both one-stage and two-stage object detection pipelines. Comprehensive experiments conducted on synthetic (i.e., noisy AI-TOD-v2.0 and DOTA-v2.0) and real-world (i.e., AI-TOD) noisy datasets demonstrate the robustness of DN-TOD under various types of label noise. Notably, when applied to the strong baseline RFLA, DN-TOD exhibits a noteworthy performance improvement of 4.9 points under 40% mixed noise.

<div align=center>
<img src="https://github.com/ZhuHaoranEIS/DN-TOD/blob/main/figures/overall.png" width="700px">
</div>

## Method
![demo image](figures/method.png)

## Installation and Get Started
[![Python](https://img.shields.io/badge/python-3.7%20tested-brightgreen)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.10.0%20tested-brightgreen)](https://pytorch.org/)

Required environments:
* Linux
* Python 3.6+
* PyTorch 1.3+
* CUDA 9.2+
* GCC 5+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
* [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)
* [RFLA](https://github.com/Chasel-Tsui/mmdet-rfla)

Install:

Note that this repository is based on the [MMDetection](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/ZhuHaoranEIS/DN-TOD.git
cd DN-TOD
pip install -r requirements/build.txt
python setup.py develop
```
## Main Contributions

- We investigate the impact of different types of label noise in tiny object detection
- We propose a DeNoising Tiny Object Detector (DNTOD) that performs robust object detection under label noise.

## Prepare datasets

- Please refer to [AI-TOD](https://github.com/Chasel-Tsui/mmdet-aitod) for AI-TOD-v2.0 and AI-TOD-v1.0 dataset.

```shell
DN-TOD
├── configs
├── data
│   ├── AI-TOD-v2
│   │   ├── annotations
│   │   │    │─── aitod_training.json
│   │   │    │─── aitod_validation.json
│   │   ├── train
│   │   │    │─── ***.png
│   │   │    │─── ***.png
│   │   ├── val
│   │   │    │─── ***.png
│   │   │    │─── ***.png
├── mmdet
```
- Generate noisy annotations:

``` shell
# generate noisy AI-TOD-v2.0 (e.g., 10%-40% all types of noise)
'''
  noise_level:
  0.1, 0.2, 0.3, 0.4 ...
  noise_type:
  first:  missing labels
  second: inaccurate bounding boxes
  third:  class shifts
  forth:  extra labels
  st:     inaccurate bounding boxes + class shifts
  all:    missing labels + inaccurate bounding boxes + class shifts + extra labels
'''

python tools/noise_generator/inject_noise.py --noise-level 0.1 0.2 0.3 0.4 \
--noise-types first second third forth st all \
--clean-path /home/zhuhaoran/DN-TOD/tools/noise_generator/clean_json/aitodv2_train.json \
--store-path /home/zhuhaoran/DN-TOD/tools/noise_generator/noisy_json

```

## Noise Influence

<div align=center>
<img src="https://github.com/ZhuHaoranEIS/DN-TOD/blob/main/figures/noise_influence.png" width="700px">
</div>

## Training

All models of DN-TOD are trained with a total batch size of 1 (can be adjusted following [MMDetection](https://github.com/open-mmlab/mmdetection)). 

- To train DN-TOD on AI-TOD-v2.0, run

```shell script
# For inaccurate bounding boxes under 10% noise
python tools/train.py inaccurate_bounding_boxes/noise_0.1.py --gpu-id 0

# For class shifts under 10% noise
python tools/train.py class_shifts/noise_0.1.py --gpu-id 0

# For mixed noise under 10% noise
python tools/train.py mixed/noise_0.1.py --gpu-id 0

```
Please refer to 
[inaccurate_bounding_boxes/noise_0.1.py](https://github.com/ZhuHaoranEIS/DN-TOD/blob/main/configs/DN-TOD-FCOS/inaccurate_bounding_boxes_noise/noise_0.1.py),  
[class_shifts/noise_0.1.py](https://github.com/ZhuHaoranEIS/DN-TOD/blob/main/configs/DN-TOD-FCOS/class_shifts_noise/noise_0.1.py), 
[mixed_shifts/noise_0.1.py](https://github.com/ZhuHaoranEIS/DN-TOD/blob/main/configs/DN-TOD-FCOS/mixed_noise/noise_0.1.py)
for model configuration

## Inference

- Modify [test.py](https://github.com/ZhuHaoranEIS/DN-TOD/blob/main/tools/test.py)

```/path/to/model_config```: modify it to the path of model config, e.g., ```.configs/DN-TOD-FCOS/inaccurate_bounding_boxes_noise/noise_0.1.py```

```/path/to/model_checkpoint```: modify it to the path of model checkpoint


- Run
```
python tools/test.py inaccurate_bounding_boxes/noise_0.1.py /path/to/model_checkpoint --eval bbox
```
