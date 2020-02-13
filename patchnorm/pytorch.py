try:
  from torch import nn
except ImportError:
  raise ImportError(f'Please install PyTorch, for example using `pip install torch`, or following the instructions at `https://pytorch.org/`.')
  

class NaivePatchNormConv2D(nn.Module):
  raise NotImplementedError('PyTorch version of NaivePatchNormConv2D')


class PatchNormConv2D(nn.Module):
  raise NotImplementedError('PyTorch version of PatchNormConv2D')
