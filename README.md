<div align="center">

# GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation

### 🎉 Accepted to ICLR 2026 🎉

[Weijia Dou](https://github.com/tj12323)<sup>1</sup>, Xu Zhang<sup>2</sup>, [Yi Bin](https://scholar.google.com/citations?user=KDdkZKQAAAAJ&hl=zh-CN&oi=sra)<sup>1*</sup>, Jian Liu<sup>3</sup>, Bo Peng<sup>2</sup>, Guoqing Wang<sup>3</sup>, Yang Yang<sup>3</sup>, [Heng Tao Shen](https://scholar.google.com.au/citations?hl=en&user=krryaDkAAAAJ)<sup>1</sup>
<small>(*Corresponding author)</small>

<sup>1</sup>Tongji University &nbsp;&nbsp; <sup>2</sup>Tianjin University &nbsp;&nbsp; <sup>3</sup>University of Electronic Science and Technology of China

[![arXiv](https://img.shields.io/badge/arXiv-2510.02186-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2510.02186)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black.svg?logo=github)](https://github.com/tj12323/GeoPurify)
[![Weights](https://img.shields.io/badge/Weights-Google_Drive-blue.svg?logo=googledrive)](https://drive.google.com/drive/folders/1eV2bbpSuQvnbr_A4tIOAh9b8Oo4a53Lb?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This is the official repository for **GeoPurify**. Our work tackles a key challenge in open-vocabulary 3D segmentation: the noisy and fragmented results produced when lifting features from 2D Vision-Language Models (VLMs) to 3D space.

GeoPurify introduces a framework that learns to **purify** these semantically-rich but geometrically-inconsistent 3D features. By distilling robust, class-agnostic geometric priors from a 3D self-supervised model, it effectively reconciles 2D semantics with 3D structure—all without needing any 3D semantic labels for its training.

> **Our key novelty in a sentence:** GeoPurify achieves state-of-the-art open-vocabulary 3D segmentation with only **~1.5% of training data** by learning to purify noisy 2D VLM features using distilled 3D geometric priors.

---

## 📝 Contents

- [GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation](#geopurify-a-data-efficient-geometric-distillation-framework-for-open-vocabulary-3d-segmentation)
  - [📝 Contents](#-contents)
  - [🧠 Method Overview](#-method-overview)
  - [✨ Key Features](#-key-features)
  - [🛠️ Installation](#️-installation)
  - [🚀 Usage](#-usage)
    - [Data Preparation](#data-preparation)
    - [Training GeoPurify](#training-geopurify)
    - [Inference](#inference)
  - [📊 Evaluation](#-evaluation)
    - [Datasets](#datasets)
    - [Metrics](#metrics)
    - [Results](#results)
  - [📦 Checkpoints](#-checkpoints)
    - [Usage](#usage)
  - [📚 Citation](#-citation)
  - [🙏 Acknowledgements](#-acknowledgements)
  - [📜 License](#-license)

---

## 🧠 Method Overview

<p align="center">
  <img src="assets/pipeline.png" alt="GeoPurify: A Data-Efficient Pipeline for Geometric Purification of 3D Semantic Features." width="700"/>
</p>

Our method explicitly decouples semantics and geometry into a two-stage pipeline:

- **Stage 1: Training (Geometric Distillation)**
  A sparse 3D Student Affinity Network (φ<sub>S</sub>) is trained to comprehend 3D structure. It learns geometric relationships directly from the point cloud by using contrastive distillation to mimic the embeddings of a powerful, frozen 3D SSL teacher (φ<sub>T</sub>, e.g., Sonata). _Crucially, this training phase requires no 3D semantic labels._
- **Stage 2: Inference (Geometry-Guided Pooling)**
  A frozen generalist 2D VLM (Ψ<sub>2D</sub>, e.g., X-Decoder) generates initial 3D features by projecting rich semantic content from multi-view images. Because these features are geometrically inconsistent, our pre-trained student network applies a geometry-aware pooling operation, using its learned affinities to iteratively refine and denoise the initial features. This process yields a final representation that is both semantically rich and geometrically coherent.

## ✨ Key Features

- **⚡ Unrivaled Data Efficiency:** Achieves or surpasses SOTA performance on major benchmarks (ScanNetV2, Matterport3D) while training on only **\~1.5%** of the data, eliminating the need for large-scale 3D annotations.
- **🎓 Novel Geometric Distillation:** Introduces a teacher-student framework that distills purely geometric affinities from a 3D self-supervised model. This learns a class-agnostic prior to correct structural inconsistencies in 2D-lifted features.
- **🌍 Strong Generalization:** The decoupled architecture provides robust zero-shot performance on long-tail benchmarks and excels in cross-dataset generalization, unlike methods that learn entangled geo-semantic representations.
- **🎯 Simple & Effective Purification:** At inference, a lightweight Geometry-Guided Pooling module uses the learned affinities to denoise features, producing coherent and accurate segmentation maps.

---

## 🛠️ Installation

For detailed setup instructions, please see the **[Installation Guide](docs/Install.md)**.

## 🚀 Usage

### Data Preparation

- **Input:** Multi-view RGB-D images + 3D point clouds.
- **Datasets supported:** [ScanNetV2](http://www.scan-net.org/), [Matterport3D](https://niessner.github.io/Matterport/), and [ScanNet200](https://kaldir.vc.in.tum.de/scannet_benchmark/).
- Follow preprocessing scripts in `scripts/preprocess`.

### Training GeoPurify

Run training with the curated subset (\~1.5% of data):

```bash
sh run/train.sh --exp_dir=out/scannet --config=config/geopurify_scannet.yaml
```

### Inference

Apply trained model for open-vocabulary 3D segmentation. Pretrained checkpoints are provided under:

- **Matterport3D:** `result/matterport/model`
- **ScanNetV2:** `result/scannet/model`

```bash
sh run/val.sh --exp_dir=out/scannet --config=config/geopurify_scannet.yaml --ckpt_name=geopurify.pth
```

---

## 📊 Evaluation

### Datasets

- **ScanNetV2:** 1,500 RGB-D scans.
- **Matterport3D:** 90 large-scale indoor scenes.
- **ScanNet200:** Long-tail benchmark emphasizing rare categories.

### Metrics

- **mIoU** (mean Intersection-over-Union)
- **mAcc** (mean Accuracy)
- **Foreground-mIoU / Foreground-mAcc** (excluding wall/floor/ceiling).

### Results

- **ScanNetV2 (∼1.5% data):** 55.1 mIoU / 72.5 mAcc
- **Matterport3D:** 40.2 mIoU / 62.4 mAcc.
- **ScanNet200 (long-tail):** 11.9 f-mIoU / 22.8 f-mAcc

---

## 📦 Checkpoints

Pretrained checkpoints are available on **Google Drive**:
🔗 [Download Here](https://drive.google.com/drive/folders/1eV2bbpSuQvnbr_A4tIOAh9b8Oo4a53Lb?usp=sharing)

### Usage

- **Matterport3D checkpoint:**
  `checkpoint/result/matterport/model/geopurify.pth`

- **ScanNetV2 checkpoint:**
  `checkpoint/result/scannet/model/geopurify.pth`

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{dou2025geopurifydataefficientgeometricdistillation,
      title={GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation},
      author={Weijia Dou and Xu Zhang and Yi Bin and Jian Liu and Bo Peng and Guoqing Wang and Yang Yang and Heng Tao Shen},
      year={2025},
      eprint={2510.02186},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.02186},
}
```

---

## 🙏 Acknowledgements

We thank the authors of [Sonata](https://github.com/facebookresearch/sonata), [X-Decoder](https://github.com/microsoft/X-Decoder), and [XMask3D](https://github.com/wangzy22/XMask3D) for their excellent open-source contributions.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
