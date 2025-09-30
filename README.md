# GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation

---

<p align="center">
  <img src="assets/pipeline.png" alt="GeoPurify: A Data-Efficient Pipeline for Geometric Purification of 3D Semantic Features." width="700"/>
</p>
<p align="center">
  <em>Our method consists of two stages. <strong>1) Training (left, dotted path):</strong> A Student Affinity Network (φ<sub>S</sub>) is trained to comprehend 3D structure. It learns geometric relationships directly from the point cloud, using contrastive distillation to mimic the embeddings of a powerful, frozen 3D SSL teacher (φ<sub>T</sub>). This training phase requires no 3D semantic labels. <strong>2) Inference (right, solid path):</strong> A frozen 2D VLM (Ψ<sub>2D</sub>) generates initial 3D features by projecting rich semantic content from multi-view images. These features, however, are geometrically inconsistent. The pre-trained student network then applies a geometry-aware pooling, using its learned affinities to refine the initial features. This process yields a final representation that is both semantically rich and geometrically coherent.</em> 
</p>

---

## 📝 Contents

* [📄 Abstract](#-abstract)
* [🛠️ Installation](#️-installation)
* [🚀 Usage](#-usage)
* [📊 Evaluation](#-evaluation)
* [📦 Checkpoints](#-checkpoints)
* [📚 Citation](#-citation)
* [🙏 Acknowledgements](#-acknowledgements)
* [📜 License](#-license)

---

## 📄 Abstract

Recent attempts to transfer features from 2D Vision–Language Models (VLMs) to 3D semantic segmentation expose a persistent trade-off. Directly projecting 2D features into 3D produces noisy and fragmented predictions, whereas enforcing geometric coherence requires costly training pipelines and large-scale annotated 3D data. We argue that this limitation stems from the dominant *segmentation-and-matching* paradigm, which fails to reconcile 2D semantics with 3D geometric structure. The geometric cues are not eliminated during the 2D-to-3D transfer but remain latent within the noisy and view-aggregated features. To exploit this property, we propose **GeoPurify** that applies a small Student Affinity Network to purify 2D VLM generated 3D point features using geometric priors distilled from a 3D self-supervised teacher model. During inference, we devise a Geometry-Guided Pooling module to further denoise the point cloud and ensure the semantic and structure consistency. Benefiting from latent geometric information and the learned affinity network, GeoPurify effectively mitigates the trade-off and achieves superior data efficiency. Extensive experiments on major 3D benchmarks show that GeoPurify matches or surpasses state-of-the-art performance while using only **\~1.5%** of the training data.

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