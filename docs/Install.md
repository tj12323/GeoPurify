# GeoPurify Installation Guide

This guide provides step-by-step instructions to set up the **GeoPurify** framework on your system.

## Prerequisites

Before beginning the installation, ensure your system meets the following requirements:

* **Python** 3.9 or higher
* **CUDA** 11.8 (for GPU support)
* **NVIDIA drivers** and **CUDA Toolkit** (for GPU usage)
* A compatible environment, such as **Ubuntu 22.04** or newer

Make sure you have **Git** and **Conda** installed. If not, install them using your system’s package manager.

## Step 1: Clone the Repository

Start by cloning the GeoPurify repository:

```bash
git clone https://github.com/tj12323/GeoPurify.git
cd GeoPurify
```

## Step 2: Initialize Submodules

GeoPurify uses submodules, so make sure to update them with the following command:

```bash
git submodule update --init --recursive
```

## Step 3: Create and Activate the Conda Environment

Create a new Conda environment with Python 3.9 and activate it:

```bash
conda create -n geopurify python=3.9 -y
conda activate geopurify
pip install setuptools==69.5.1
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

To efficiently perform sparse tensor operations, install **MinkowskiEngine** with the following commands:

```bash
pip install numpy==1.26.4
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps -v
```

## Step 6: Install Additional Libraries

Install other required dependencies for GeoPurify:

```bash
pip install torch-scatter==2.1.2
pip install packaging==24.2
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install scipy==1.7.3
```

## Step 7: Set Up Third-Party Dependencies

GeoPurify relies on **Sonata** and **X-Decoder**, which are located in the `third_party/` directory. Follow the setup instructions for each:

### Sonata

Navigate to the `sonata` directory and install it:

```bash
cd third_party/sonata
pip install spconv-cu118
pip install -e .
```

### X-Decoder

Navigate to the `X-Decoder` directory and install the required dependencies:

```bash
cd ../X-Decoder
conda install -c conda-forge mpi4py
pip install -r ../../docs/requirements.txt
pip install -e .
```

Download the required model checkpoint:

```bash
cd xdecoder/checkpoint
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/focall_vision_focalb_lang_unicl.pt
```

Then, install **Detectron2**:

```bash
cd ../detectron2
python -m pip install -e .
```

## Final Steps

After setting up **Sonata** and **X-Decoder**, navigate back to the root directory and install GeoPurify:

```bash
cd ../../..
pip install -e .
```

Finally, install **Faiss** (for fast similarity search):

```bash
conda install -c conda-forge faiss-cpu
```