# PatchNorm

This contains the EfficientPatchNormConv2D and PatchNormConv2D layers.

![patch_norm](patch_norm.jpg "patch_norm")

## Installation

Over ssh:
```bash
python3 -m pip install -U git+ssh://git@github.com/benjamindkilleen/patchnorm.git --user
```

Or over https:
```bash
python3 -m pip install -U git+https://github.com/benjamindkilleen/efficientnet_pn.git --user
```

## Usage

```python
from patchnorm import PatchNormConv2D, EfficientPatchNormConv2D
```
