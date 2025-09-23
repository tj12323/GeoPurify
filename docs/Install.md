# GeoPurify Installation Guide

This document provides step-by-step instructions to set up the **GeoPurify** framework on your system.

## Prerequisites

* **Python** 3.9 or higher
* **CUDA** 11.8 (for GPU support)
* **NVIDIA drivers** and **CUDA Toolkit** installed (for GPU usage)
* A compatible environment (e.g., **Ubuntu 22.04** or newer)

Ensure you have **Git** and **Conda** installed. If not, install them as per your system’s package manager.

## Step 1: Clone the Repository

Start by cloning the GeoPurify repository:

```bash
cd GeoPurify
```

## Step 2: Initialize Submodules

GeoPurify uses submodules, so update them with the following command:

```bash
git submodule update --init --recursive
```

## Step 3: Create and Activate the Conda Environment

Create a new Conda environment with Python 3.9 and activate it:

```bash
conda create -n geopurify python=3.9 -y
conda activate geopurify
```

## Step 4: Install PyTorch and Dependencies

Install PyTorch, TorchVision, and Torchaudio with CUDA 11.8 support (necessary for GPU acceleration):

```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

Next, install **Ninja** (a build tool) and **libopenblas-dev** (for optimized matrix operations):

```bash
pip install ninja
sudo apt install libopenblas-dev
```

## Step 5: Install MinkowskiEngine

Install **MinkowskiEngine** (for efficient sparse tensor operations), ensuring that the installation uses the correct dependencies:

```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps -v
```

## Step 6: Install Other Required Libraries

Install additional dependencies for GeoPurify:

```bash
pip install torch-scatter==2.1.2+pt25cu118
pip install git+https://github.com/Dao-AILab/flash-attention.git@2.7.4.post1
pip install scipy==1.7.3
pip install numpy==1.26.4
```

## Step 7: Set Up Third-Party Dependencies

GeoPurify relies on **Sonata** and **X-Decoder**, which are located in the `third_party/` directory. Follow the respective setup instructions:

* **Sonata:**
  Navigate to the `sonata` directory and install it:

  ```bash
  cd third_party/sonata
  pip install -e .
  ```

* **X-Decoder:**
  Navigate to the `X-Decoder` directory and install it:

  ```bash
  cd ../X-Decoder
  pip install -e .
  ```

**Note:** Follow any additional setup instructions in their respective `README.md` files (`third_party/sonata/README.md` and `third_party/X-Decoder/xdecoder/README.md`).
