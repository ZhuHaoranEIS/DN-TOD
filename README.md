# Robust Tiny Object Detection in Aerial Images amidst Label Noise
This is the official implementation of the paper "Robust Tiny Object Detection in Aerial Images amidst Label Noise". [arxiv](https://arxiv.org/abs/2401.08056)

## Introduction
DN-TOD is an effective denoising algorithm that can be integrated into either one-stage or two-stage algorithms to enhance the network's robustness against noise.

**Abstract**: Precise detection of tiny objects in remote sensing imagery remains a significant challenge due to their limited visual information and frequent occurrence within scenes. This challenge is further exacerbated by the practical burden and inherent errors associated with manual annotation: annotating tiny objects is laborious and prone to errors (i.e., label noise). Training detectors for such objects using noisy labels often leads to suboptimal performance, with networks tending to overfit on noisy labels. In this study, we address the intricate issue of tiny object detection under noisy label supervision. We systematically investigate the impact of various types of noise on network training, revealing the vulnerability of object detectors to class shifts and inaccurate bounding boxes for tiny objects. To mitigate these challenges, we propose a DeNoising Tiny Object Detector (DN-TOD), which incorporates a Class-aware Label Correction (CLC) scheme to address class shifts and a Trend-guided Learning Strategy (TLS) to handle bounding box noise. CLC mitigates inaccurate class supervision by identifying and filtering out class-shifted positive samples, while TLS reduces noisy box-induced erroneous supervision through sample reweighting and bounding box regeneration. Additionally, Our method can be seamlessly integrated into both one-stage and two-stage object detection pipelines. Comprehensive experiments conducted on synthetic (i.e., noisy AI-TOD-v2.0 and DOTA-v2.0) and real-world (i.e., AI-TOD) noisy datasets demonstrate the robustness of DN-TOD under various types of label noise. Notably, when applied to the strong baseline RFLA, DN-TOD exhibits a noteworthy performance improvement of 4.9 points under 40% mixed noise.

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
git clone xxx
cd xxx
pip install -r requirements/build.txt
python setup.py develop
```
## Prepare datasets

Please refer to [AI-TOD-v2.0](https://github.com/Chasel-Tsui/mmdet-aitod) for AI-TOD-v2.0 and AI-TOD-v1.0 dataset.

```shell
DN-TOD
├── mmdet
├── tools
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


