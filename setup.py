import os
from setuptools import find_packages, setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 13], "Requires PyTorch >= 1.13.1"


setup(
    name="geopurify",
    author="Weijia Dou",
    url="https://github.com/tj12323/GeoPurify.git",
    description="A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation",
    python_requires=">=3.8",
    packages=find_packages(exclude=("config")),
    install_requires=[
    ],
    include_package_data=True,
)
