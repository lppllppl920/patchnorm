try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import math

except ImportError:
  raise ImportError(
    f'Please install PyTorch, for example using `pip install torch`, or following the instructions at `https://pytorch.org/`.')

from . import utils


class NaivePatchNormConv2D(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      bias=True,
      epsilon=1.0e-3,
      **kwargs):
    """Naively implemented patch-normalized convolution.

    :param in_channels: number of channels in the input image
    :param out_channels: number channels in the output image
    :param kernel_size: size of the kernel
    :param stride: stride of the convolution
    :param padding: amount of implicit zero padding
    :param dilation: dilation amount
    :param bias: whether to use a bias vector
    :param epsilon: value for epsilon

    """
    super(NaivePatchNormConv2D, self).__init__(**kwargs)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = utils.tuplify(kernel_size, 2)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    
    self.beta = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)
    self.gamma = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)
    self.conv = nn.Conv2d(
      in_channels=self.in_channels,
      out_channels=self.out_channels,
      kernel_size=self.kernel_size,
      stride=self.kernel_size,
      padding=0,
      bias=bias)
    self.epsilon = epsilon
    
  @staticmethod
  def extract_image_patches(x, kernel_size, stride=1, dilation=1):
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel_size[0] - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel_size[1] - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2), mode='constant',
              value=0)
    unfold = nn.Unfold(kernel_size=kernel_size)
    patches = unfold(x)
    return patches.view(b, kernel_size[0] * kernel_size[1] * c, h2, w2)

  def forward(self, x):
    # B x k^2*C x H` x W`
    patches = self.extract_image_patches(x, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
    b, k2c, h_out, w_out = patches.shape
    patches_mean = torch.mean(patches, dim=1, keepdim=True)
    patches_std = torch.std(patches, dim=1, keepdim=True, unbiased=True)
    centered = (patches - patches_mean) / (patches_std + self.epsilon)
    # B x k x k x C x H` x W`
    centered = centered.view(-1, self.kernel_size[0], self.kernel_size[0], x.shape[1], centered.shape[2],
                             centered.shape[3])
    shifted = self.gamma.view(1, 1, 1, -1, 1, 1) * centered + self.beta.view(1, 1, 1, -1, 1, 1)
    # B x C x H` x k x W` x k
    shifted = shifted.permute(0, 3, 4, 1, 5, 2).contiguous()
    # B x C x k*H' x k*W'
    shifted = shifted.view(-1, x.shape[1], self.kernel_size[0] * h_out, self.kernel_size[1] * w_out)
    return self.conv(shifted)

  def freeze_weights_except_norm(self):
    """Freeze the weights in this layer except for beta and gamma.
    """
    for param in self.conv.parameters():
      param.requires_grad = False

  def unfreeze_weights(self):
    for param in self.conv.parameters():
      param.requires_grad = True
      

class PatchNormConv2D(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      use_bias=True,
      epsilon=1.0e-3,
      **kwargs):
    """Efficiently implemented patch-normalized convolution.

    :param in_channels: number of channels in the input image
    :param out_channels: number channels in the output image
    :param kernel_size: size of the kernel
    :param stride: stride of the convolution
    :param padding: amount of implicit zero padding
    :param dilation: dilation amount
    :param bias: whether to use a bias vector
    :param epsilon: value for epsilon

    """
    super(PatchNormConv2D, self).__init__(**kwargs)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = utils.tuplify(kernel_size, 2)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.use_bias = use_bias

    self.beta = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)
    self.gamma = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)

    self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                          kernel_size=self.kernel_size,
                          stride=self.stride, padding=self.padding, bias=False)
    self.epsilon = epsilon
    self.window = (1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[0]) * torch.ones(1).view(1, 1, 1, 1)
                   .expand(1, in_channels, self.kernel_size[0], self.kernel_size[1]).contiguous().cuda())

    if self.use_bias:
        self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
      # B x 1 x H' x W'
      means = F.conv2d(x, self.window, bias=None, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=1)
      square_means = F.conv2d(x ** 2, self.window, bias=None, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=1)
      stds = torch.sqrt(square_means - means ** 2 + self.epsilon)
      
      # B x C_out x H' x W'
      conv = self.conv(self.gamma.view(1, -1, 1, 1) * x) / stds
      
      # 1 x C_out x 1 x 1
      gamma_kernel_sum = torch.sum(self.conv.weight * self.gamma.view(1, -1, 1, 1), dim=(1, 2, 3), keepdim=True)
      gamma_kernel_sum = gamma_kernel_sum.view(1, -1, 1, 1)

      # B x C_out x H' x W'
      kernel_weighted_means = means * gamma_kernel_sum / stds
      
      # 1 x C_out x 1 x 1
      beta_kernel_sum = torch.sum(self.conv.weight * self.beta.view(1, -1, 1, 1), dim=(1, 2, 3), keepdim=True)
      beta_kernel_sum = beta_kernel_sum.view(1, -1, 1, 1)

      # B x C_out x H' x W'
      x = conv - kernel_weighted_means + beta_kernel_sum
      
      if self.use_bias:
        x = x + self.bias.view(1, -1, 1, 1)

      return x

    def freeze_weights_except_norm(self):
      """Freeze the weights in this layer except for beta and gamma.
      """
      for param in self.conv.parameters():
        param.requires_grad = False
      if self.bias:
        for param in self.bias.parameters():
          param.requires_grad = False
          
    def unfreeze_weights(self):
      for param in self.conv.parameters():
        param.requires_grad = True
      if self.bias:
        for param in self.bias.parameters():
          param.requires_grad = True


class PatchNormConv3D(nn.Module):
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size,
      stride=1,
      padding=0,
      dilation=1,
      use_bias=True,
      epsilon=1.0e-3,
      **kwargs):
    """Efficiently implemented patch-normalized convolution for 3D data.

    :param in_channels: number of channels in the input image
    :param out_channels: number channels in the output image
    :param kernel_size: size of the kernel
    :param stride: stride of the convolution
    :param padding: amount of implicit zero padding
    :param dilation: dilation amount
    :param bias: whether to use a bias vector
    :param epsilon: value for epsilon

    """
    super(PatchNormConv2D, self).__init__(**kwargs)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = utils.tuplify(kernel_size, 3)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.use_bias = use_bias

    self.beta = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)
    self.gamma = nn.Parameter(torch.ones(self.in_channels, dtype=torch.float32), requires_grad=True)

    self.conv = nn.Conv3d(
      in_channels=self.in_channels,
      out_channels=self.out_channels,
      kernel_size=self.kernel_size,
      stride=self.stride,
      padding=self.padding,
      bias=False)
    self.epsilon = epsilon
    self.window = (1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]) * torch.ones(1)
                   .view(1, 1, 1, 1, 1).expand(1, in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
                   .contiguous().cuda())

    if self.use_bias:
        self.bias = nn.Parameter(torch.zeros(self.out_channels, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
      # B x 1 x H' x W'
      means = F.conv3d(x, self.window, bias=None, stride=self.stride,
                       padding=self.padding, dilation=self.dilation, groups=1)
      square_means = F.conv3d(x ** 2, self.window, bias=None, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=1)
      stds = torch.sqrt(square_means - means ** 2 + self.epsilon)
      
      # B x C_out x H' x W'
      conv = self.conv(self.gamma.view(1, -1, 1, 1, 1) * x) / stds
      
      # 1 x C_out x 1 x 1
      gamma_kernel_sum = torch.sum(self.conv.weight * self.gamma.view(1, -1, 1, 1, 1), dim=(1, 2, 3, 4), keepdim=True)
      gamma_kernel_sum = gamma_kernel_sum.view(1, -1, 1, 1, 1)

      # B x C_out x H' x W'
      kernel_weighted_means = means * gamma_kernel_sum / stds
      
      # 1 x C_out x 1 x 1
      beta_kernel_sum = torch.sum(self.conv.weight * self.beta.view(1, -1, 1, 1, 1), dim=(1, 2, 3, 4), keepdim=True)
      beta_kernel_sum = beta_kernel_sum.view(1, -1, 1, 1, 1)

      # B x C_out x H' x W'
      x = conv - kernel_weighted_means + beta_kernel_sum
      
      if self.use_bias:
        x = x + self.bias.view(1, -1, 1, 1, 1)

      return x

    def freeze_weights_except_norm(self):
      """Freeze the weights in this layer except for beta and gamma.
      """
      for param in self.conv.parameters():
        param.requires_grad = False
      if self.bias:
        for param in self.bias.parameters():
          param.requires_grad = False
          
    def unfreeze_weights(self):
      for param in self.conv.parameters():
        param.requires_grad = True
      if self.bias:
        for param in self.bias.parameters():
          param.requires_grad = True

          
def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal", gpu_id=0):
  assert (torch.cuda.is_available())
  net = net.cuda(gpu_id)
  if type == "glorot":
    glorot_weight_zero_bias(net, distribution=distribution)
  else:
    kaiming_weight_zero_bias(net, mode=mode, activation_mode=activation_mode, distribution=distribution)
  return net


def glorot_weight_zero_bias(model, distribution="uniform"):
  """
  Initalize parameters of all modules
  by initializing weights with glorot  uniform/xavier initialization,
  and setting biases to zero.
  Weights from batch norm layers are set to 1.

  Parameters
  ----------
  model: Module
  distribution: string
  """
  for module in model.modules():
    if hasattr(module, 'weight'):
      if not ('BatchNorm' in module.__class__.__name__):
        if distribution == "uniform":
          torch.nn.init.xavier_uniform_(module.weight, gain=1)
        else:
          torch.nn.init.xavier_normal_(module.weight, gain=1)
      else:
        torch.nn.init.constant_(module.weight, 1)
    if hasattr(module, 'bias'):
      if module.bias is not None:
        torch.nn.init.constant_(module.bias, 0)


def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
  if activation_mode == "leaky_relu":
    print("Leaky relu is not supported yet")
    assert False

  for module in model.modules():
    if hasattr(module, 'weight'):
      if not ('BatchNorm' in module.__class__.__name__):
        if distribution == "uniform":
          torch.nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=activation_mode)
        else:
          torch.nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=activation_mode)
      else:
        torch.nn.init.constant_(module.weight, 1)
    if hasattr(module, 'bias'):
      if module.bias is not None:
        torch.nn.init.constant_(module.bias, 0)


# Unit test
if __name__ == "__main__":
  import numpy as np
  import random

  # Fix randomness for reproducibility
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

  for i in range(100):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(1)

    test_image = np.random.randn(1, 1, 200, 200).astype(np.float32)
    # test_image = np.ones((1, 1, 200, 200), dtype=np.float32)
    # test_image = np.repeat(np.linspace(start=1, stop=200, num=200).reshape((1, 200)), repeats=200, axis=0).reshape(
    #     (1, 1, 200, 200)).astype(np.float32)
    test_image = torch.from_numpy(test_image).cuda()

    patch_norm_conv = PatchNormConv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1,
                                      dilation=1)
    patch_norm_conv = init_net(patch_norm_conv)
    output_image = patch_norm_conv(test_image)

    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(1)
    patch_norm_conv_2 = NaivePatchNormConv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1,
                                             dilation=1)
    patch_norm_conv_2 = init_net(patch_norm_conv_2)
    output_image_2 = patch_norm_conv_2(test_image)

    loss = torch.mean(output_image)
    loss_2 = torch.mean(output_image_2)

    loss.backward()
    loss_2.backward()

    print(torch.mean(patch_norm_conv.conv.weight.grad - patch_norm_conv_2.conv.weight.grad))
    print(torch.mean(patch_norm_conv.beta.grad - patch_norm_conv_2.beta.grad))
    print(torch.mean(patch_norm_conv.gamma.grad - patch_norm_conv_2.gamma.grad))
    print(torch.mean(output_image - output_image_2))
