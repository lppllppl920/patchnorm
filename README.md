# Patch-normalized Convolution

![patch_norm](patch_norm.jpg "patch_norm")

Implementation of patch-normalized convolutional layers in PyTorch and Tensorflow Keras.

## Installation

Over ssh:
```bash
python3 -m pip install -U git+ssh://git@github.com/benjamindkilleen/patchnorm.git --user
```

Or over https:
```bash
python3 -m pip install -U git+https://github.com/benjamindkilleen/patchnorm.git --user
```

## Usage

This package contains both Tensorflow Keras and PyTorch implementations of the PatchNorm layers.
They can be imported independently of one another, for example:
```python
from patchnorm.tfkeras import PatchNormConv2D
from patchnorm.pytorch import PatchNormConv2D
```
