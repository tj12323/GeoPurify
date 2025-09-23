# GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation

---

<p align="center">
  <img src="assets/pipleine.png" alt="GeoPurify: A Data-Efficient Pipeline for Geometric Purification of 3D Semantic Features." width="700"/>
</p>
<p align="center">
  <em>Our method consists of two stages. \textbf{1) Training (left, dotted path):} A Student Affinity Network ($\phi_S$) is trained to comprehend 3D structure. It learns geometric relationships directly from the point cloud, using contrastive distillation to mimic the embeddings of a powerful, frozen 3D SSL teacher ($\phi_T$). This training phase requires no 3D semantic labels. \textbf{2) Inference (right, solid path):} A frozen 2D VLM ($\Psi_{2D}$) generates initial 3D features by projecting rich semantic content from multi-view images. These features, however, are geometrically inconsistent. The pre-trained student network then applies a geometry-aware pooling, using its learned affinities to refine the initial features. This process yields a final representation that is both semantically rich and geometrically coherent.</em>
</p>

---

## 📝 Contents

* [📄 Abstract](#-abstract)
* [🛠️ Installation](#️-installation)
* [🚀 Usage](#-usage)

  * [Data Preparation](#data-preparation)
  * [Training GeoPurify](#training-geopurify)
  * [Inference](#inference)
* [📊 Evaluation](#-evaluation)

  * [Datasets](#datasets)
  * [Metrics](#metrics)
  * [Results](#results)

---

## 📄 Abstract

Transferring features from 2D Vision-Language Models (VLMs) for 3D understanding forces an unfavorable trade-off: methods must either accept the noisy, fragmented outputs of direct projection or engage in computationally expensive training to learn geometric coherence. This conflict stems from the prevailing "Segmentation and Matching" paradigm, which struggles to reconcile 2D semantics with 3D structure. Empirical evidence suggests that geometric information is not destroyed during this 2D-to-3D transfer but is instead rendered latent within the noisy, view-aggregated features. To recover this structure, we introduce **GeoPurify**, a data-efficient teacher-student framework that purifies VLM features using geometric priors from a 3D self-supervised model. GeoPurify employs Geometric Contrastive Distillation to train a student module on latent structural relationships, guided by the teacher's priors. At inference, a Geometry-Guided Pooling module leverages these learned affinities to denoise features and enforce spatial consistency. Extensive experiments on major 3D benchmarks demonstrate that GeoPurify achieves performance comparable to or surpassing state-of-the-art methods. Crucially, it resolves the semantics-geometry trade-off in a data-efficient manner, requiring no manual annotations. Our code and models will be made publicly available.

---

## 🛠️ Installation

For detailed setup instructions, please see the **[Installation Guide](docs/Install.md)**.

---

## 🚀 Usage

### Data Preparation

* **Input:** Multi-view RGB-D images + 3D point clouds.
* **Datasets supported:** [ScanNetV2](http://www.scan-net.org/), [Matterport3D](https://niessner.github.io/Matterport/), and [ScanNet200](https://kaldir.vc.in.tum.de/scannet_benchmark/).
* Follow preprocessing scripts in `scripts/preprocess`.

### Training GeoPurify

Run training with the curated subset (\~1.5% of data):

```bash
sh run/train.sh --exp_dir=xdecoder_test/out/scannet --config=config/geopurify_scannet.yaml
```

### Inference

Apply trained model for open-vocabulary 3D segmentation:

```bash
sh run/val.sh --exp_dir=xdecoder_test/out/scannet --config=config/geopurify_scannet.yaml --ckpt_name=affinity_predictor_epoch_34.pth
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
