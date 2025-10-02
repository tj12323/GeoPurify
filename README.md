# GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation

This is the official repository for **GeoPurify**. Our work tackles a key challenge in open-vocabulary 3D segmentation: the noisy and fragmented results produced when lifting features from 2D Vision-Language Models (VLMs) to 3D space.

GeoPurify introduces a framework that learns to **purify** these semantically-rich but geometrically-inconsistent 3D features. By distilling robust, class-agnostic geometric priors from a 3D self-supervised model, it effectively reconciles 2D semantics with 3D structure—all without needing any 3D semantic labels for its training.

> **Our key novelty in a sentence:** GeoPurify achieves state-of-the-art open-vocabulary 3D segmentation with only **\~1.5% of training data** by learning to purify noisy 2D VLM features using distilled 3D geometric priors.

<p align="center">
  <img src="assets/pipeline.png" alt="GeoPurify: A Data-Efficient Pipeline for Geometric Purification of 3D Semantic Features." width="700"/>
</p>
<p align="center">
  <em>Our method consists of two stages. <strong>1) Training (left, dotted path):</strong> A Student Affinity Network (φ<sub>S</sub>) is trained to comprehend 3D structure. It learns geometric relationships directly from the point cloud, using contrastive distillation to mimic the embeddings of a powerful, frozen 3D SSL teacher (φ<sub>T</sub>). This training phase requires no 3D semantic labels. <strong>2) Inference (right, solid path):</strong> A frozen 2D VLM (Ψ<sub>2D</sub>) generates initial 3D features by projecting rich semantic content from multi-view images. These features, however, are geometrically inconsistent. The pre-trained student network then applies a geometry-aware pooling, using its learned affinities to refine the initial features. This process yields a final representation that is both semantically rich and geometrically coherent.</em> 
</p>

---

## 📝 Contents

* [✨ Key Features](#-key-features)
* [🛠️ Installation](#️-installation)
* [🚀 Usage](#-usage)
* [📊 Evaluation](#-evaluation)
* [📦 Checkpoints](#-checkpoints)
* [📚 Citation](#-citation)
* [🙏 Acknowledgements](#-acknowledgements)
* [📜 License](#-license)

---

## ✨ Key Features

  * **⚡ Unrivaled Data Efficiency:** Achieves or surpasses SOTA performance on major benchmarks (ScanNetV2, Matterport3D) while training on only **\~1.5%** of the data, eliminating the need for large-scale 3D annotations.
  * **🎓 Novel Geometric Distillation:** Introduces a teacher-student framework that distills purely geometric affinities from a 3D self-supervised model. This learns a class-agnostic prior to correct structural inconsistencies in 2D-lifted features.
  * **🌍 Strong Generalization:** The decoupled architecture provides robust zero-shot performance on long-tail benchmarks and excels in cross-dataset generalization, unlike methods that learn entangled geo-semantic representations.
  * **🎯 Simple & Effective Purification:** At inference, a lightweight Geometry-Guided Pooling module uses the learned affinities to denoise features, producing coherent and accurate segmentation maps.


---

## 🛠️ Installation

For detailed setup instructions, please see the **[Installation Guide](docs/Install.md)**.

## 🚀 Usage

### Data Preparation

* **Input:** Multi-view RGB-D images + 3D point clouds.
* **Datasets supported:** [ScanNetV2](http://www.scan-net.org/), [Matterport3D](https://niessner.github.io/Matterport/), and [ScanNet200](https://kaldir.vc.in.tum.de/scannet_benchmark/).
* Follow preprocessing scripts in `scripts/preprocess`.

### Training GeoPurify

Run training with the curated subset (\~1.5% of data):

```bash
sh run/train.sh --exp_dir=out/scannet --config=config/geopurify_scannet.yaml
```

### Inference

Apply trained model for open-vocabulary 3D segmentation. Pretrained checkpoints are provided under:

* **Matterport3D:** `result/matterport/model`
* **ScanNetV2:** `result/scannet/model`

```bash
sh run/val.sh --exp_dir=out/scannet --config=config/geopurify_scannet.yaml --ckpt_name=geopurify.pth
```

---

## 📊 Evaluation

### Datasets

* **ScanNetV2:** 1,500 RGB-D scans.
* **Matterport3D:** 90 large-scale indoor scenes.
* **ScanNet200:** Long-tail benchmark emphasizing rare categories.

### Metrics

* **mIoU** (mean Intersection-over-Union)
* **mAcc** (mean Accuracy)
* **Foreground-mIoU / Foreground-mAcc** (excluding wall/floor/ceiling).

### Results

* **ScanNetV2 (∼1.5% data):** 55.1 mIoU / 72.5 mAcc
* **Matterport3D:** 40.2 mIoU / 62.4 mAcc.
* **ScanNet200 (long-tail):** 11.9 f-mIoU / 22.8 f-mAcc

---

## 📦 Checkpoints

Pretrained checkpoints are available on **Google Drive**:
🔗 [Download Here](https://drive.google.com/drive/folders/1eV2bbpSuQvnbr_A4tIOAh9b8Oo4a53Lb?usp=sharing)

### Usage

* **Matterport3D checkpoint:**
  `checkpoint/result/matterport/model/geopurify.pth`

* **ScanNetV2 checkpoint:**
  `checkpoint/result/scannet/model/geopurify.pth`


## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{your2025geopurify,
  title   = {GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation},
  author  = {Author, A. and Author, B. and Author, C.},
  journal = {arXiv preprint arXiv:2501.12345},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

We thank the authors of [Sonata](https://github.com/facebookresearch/sonata), [X-Decoder](https://github.com/microsoft/X-Decoder), and [XMask3D](https://github.com/wangzy22/XMask3D) for their excellent open-source contributions.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).