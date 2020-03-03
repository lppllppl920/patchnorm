try:
  import tensorflow as tf
  from tensorflow import keras
except ImportError:
  raise ImportError(f'Please install Tensorflow, for example using `pip install tensorflow`, or following the instructions at `https://www.tensorflow.org`.')
  
from . import utils


logger = tf.get_logger()


class NaivePatchNormConv2D(keras.layers.Layer):
  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               patch_size=None,
               epsilon=0.00001,
               channel_wise=False,
               use_nu=False,
               channel_wise_nu=False,
               simple=None,
               axis=None,
               **kwargs):
    """Patch-normalized convolution using extract_patches.

    This is a memory inefficient implementation of the patch-normalized convolution and is included for edification. From the Tensorflow docs:

    :param filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution)
    :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
    :param strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    :param padding: one of `"valid"` or `"same"` (case-insensitive).
    :param activation: 
    :param use_bias: 
    :param kernel_initializer: 
    :param bias_initializer: 
    :param kernel_regularizer: 
    :param bias_regularizer: 
    :param activity_regularizer: 
    :param kernel_constraint: 
    :param bias_constraint: 
    :param patch_size: 
    :param epsilon: value of epsilon to use.
    :param channel_wise: use per-channel beta and gamma parameters.

    """
    super().__init__(**kwargs)

    self.filters = filters
    self.kernel_size = utils.tuplify(kernel_size, 2)
    self.strides = utils.tuplify(strides, 2)
    self.padding = padding
    self.activation = activation
    self.dilation_rate = dilation_rate
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.axis = 3
    self.epsilon = epsilon
    self.channel_wise = channel_wise if simple is None else not simple
    self.use_nu = use_nu
    self.channel_wise_nu = channel_wise_nu
    self.simple = simple
    self.axis = axis

    self.patch_size = self._get_receptive_field() if patch_size is None else utils.tuplify(patch_size, 2)

    assert axis is None or axis == 3, 'axis == 3 or None'
    assert self.padding == 'same' or self.kernel_size == (1, 1), 'todo: padding != same, especially for patch_size != kernel_size'
    if self.kernel_size == (1, 1):
      self.padding = 'same'

    if self.dilation_rate != 1:
      assert self.padding.lower() == 'same'
      assert self.strides == (1, 1)

  def build(self, input_shape):
    self.beta = self.add_weight('beta',
                                shape=(input_shape[3],) if self.channel_wise else (1,),
                                dtype=self.dtype,
                                trainable=True,
                                initializer='zeros')
    self.gamma = self.add_weight('gamma',
                                 shape=(input_shape[3],) if self.channel_wise else (1,),
                                 dtype=self.dtype,
                                 trainable=True,
                                 initializer='ones')
    if self.use_nu:
      # could be (C,), but only in the inefficient implementation, otherwise has to be (filters,)
      self.nu = self.add_weight('nu',
                                shape=(input_shape[3],) if self.channel_wise_nu else (1,),
                                dtype=self.dtype,
                                trainable=True,
                                initializer='zeros')

    self.conv = keras.layers.Conv2D(
      filters=self.filters,
      kernel_size=self.kernel_size,
      strides=self.kernel_size,
      padding='valid',
      dilation_rate=self.dilation_rate,
      activation=self.activation,
      use_bias=self.use_bias,
      kernel_initializer=self.kernel_initializer,
      bias_initializer=self.bias_initializer,
      kernel_regularizer=self.kernel_regularizer,
      bias_regularizer=self.bias_regularizer,
      activity_regularizer=self.activity_regularizer,
      kernel_constraint=self.kernel_constraint,
      bias_constraint=self.bias_constraint)

  def _get_receptive_field(self):
    if self.dilation_rate is None or self.dilation_rate == 1:
      return self.kernel_size
    else:
      return (self.dilation_rate * (self.kernel_size[0] + 1) - 1,
              self.dilation_rate * (self.kernel_size[1] + 1) - 1)
    
  def call(self, x):
    # (N, H', W', h * w * C)
    patches = tf.image.extract_patches(
      images=x,
      sizes=[1, self.patch_size[0], self.patch_size[1], 1],
      strides=[1, self.strides[0], self.strides[1], 1],
      rates=[1, 1, 1, 1],
      padding=self.padding.upper())

    # (N, H', W', h, w, C)
    patches = tf.reshape(patches, (-1, patches.shape[1], patches.shape[2], self.kernel_size[0], self.kernel_size[1], x.shape[3]))
    
    # (N, H', W', 1, 1, 1)
    means = tf.math.reduce_mean(patches, axis=(3, 4, 5), keepdims=True)
    stds = tf.math.reduce_std(patches, axis=(3, 4, 5), keepdims=True)

    # (N, H', W', 1, 1, 1) or (N, H', W', 1, 1, C) if channel_wise_nu
    if self.use_nu:
      nu = tf.reshape(self.nu, (1, 1, 1, -1))
      centered = (patches - means) / tf.sqrt(tf.square(stds) + tf.square(nu) + self.epsilon)
    else:
      centered = (patches - means) / tf.sqrt(tf.square(stds) + self.epsilon)
    
    shifted = tf.reshape(self.gamma, (1, 1, 1, 1, 1, -1)) * centered + tf.reshape(self.beta, (1, 1, 1, 1, 1, -1))

    # (N, H', h, W', w, C)
    shifted = tf.transpose(shifted, perm=[0, 1, 3, 2, 4, 5])
    shifted = tf.reshape(shifted, [-1, shifted.shape[1] * shifted.shape[2], shifted.shape[3] * shifted.shape[4], shifted.shape[5]])

    # (N, H', W', filters)
    return self.conv(shifted)

  def get_config(self):
    config = super().get_config()
    config.update({'filters': self.filters,
                   'kernel_size': self.kernel_size,
                   'strides': self.strides,
                   'padding': self.padding,
                   'activation': self.activation,
                   'use_bias': self.use_bias,
                   'kernel_initializer': self.kernel_initializer,
                   'bias_initializer': self.bias_initializer,
                   'kernel_regularizer': self.kernel_regularizer,
                   'bias_regularizer': self.bias_regularizer,
                   'activity_regularizer': self.activity_regularizer,
                   'kernel_constraint': self.kernel_constraint,
                   'bias_constraint': self.bias_constraint,
                   'axis': self.axis,
                   'patch_size': self.patch_size,
                   'channel_wise': self.channel_wise,
                   'simple': self.simple})
    return config

  def set_weights_from_conv(self, weights):
    """Set the weights as from a convolutional layer.

    :param weights:
    :returns:
    :rtype:

    """
    self.conv.set_weights(weights)

  def set_weights_from_bn(self, weights, silent=False):
    """Sets beta and gamma from `weights` as returned by a BatchNormalization layer.

    Note that the BatchNormalization includes a running mean and variance which PatchNorm does not use. These are ignored.

    Note also that this only works when beta and gamma have `filters` elements. If this is not the case, this function prints a warning.

    Assumes that the get_weights() function has returned the list with [beta, gamma, moving_mean, moving_variance].

    :param weights: 
    :param silent: ignore warnings
    :returns: 
    :rtype: 

    """
    gamma = weights[0]
    beta = weights[1]

    if gamma.shape != self.gamma.shape:
      if not silent:
        logger.warning('shape mismatch between gamma and self.gamma: {} != {}'.format(gamma.shape, self.gamma.shape))
    elif beta.shape != self.beta.shape:
      if not silent:
        logger.warning('shape mismatch between beta and self.beta: {} != {}'.format(beta.shape, self.beta.shape))
    else:
      self.gamma.assign(gamma)
      self.beta.assign(beta)
      logger.info('successfully set pn beta + gamma weights')

  def freeze_weights_except_norm(self):
    """Freeze the weights in this layer except for beta and gamma.
    """
    self.conv.trainable = False
    
  def unfreeze_weights(self):
    self.conv.trainable = True


class BiasAdd(keras.layers.Layer):
  """A simple layer that adds a bias along the channel dimension.

  Included so as to make bias weights non-trainable for EfficientPatchNormConv2D.

  """
  def __init__(self, initializer=None, regularizer=None, constraint=None, **kwargs):
    super().__init__(**kwargs)
    self.initializer = initializer
    self.regularizer = regularizer
    self.constraint = constraint

  def build(self, input_shape):
    self.bias = self.add_weight(
      'bias',
      shape=(input_shape[3],),
      initializer=self.initializer,
      regularizer=self.regularizer,
      constraint=self.constraint,
      trainable=True,
      dtype=self.dtype)

  def call(self, x):
    x += tf.reshape(self.bias, (1, 1, 1, -1))
    return x

  def get_config(self):
    config = super().get_config()
    config.update({'initializer': self.initializer,
                   'regularizer': self.regularizer,
                   'constraint': self.constraint})
    return config

  
class PatchNormConv2D(NaivePatchNormConv2D):
  def build(self, input_shape):
    self.beta = self.add_weight('beta',
                                shape=(input_shape[3],) if self.channel_wise else (1,),
                                dtype=self.dtype,
                                trainable=True,
                                initializer='zeros')
    self.gamma = self.add_weight('gamma',
                                 shape=(input_shape[3],) if self.channel_wise else (1,),
                                 dtype=self.dtype,
                                 trainable=True,
                                 initializer='ones')
    if self.use_nu:
      # could be (C,), but only in the inefficient implementation, otherwise has to be (filters,)
      self.nu = self.add_weight('nu',
                                shape=(self.filters,) if self.channel_wise_nu else (1,),
                                dtype=self.dtype,
                                trainable=True,
                                initializer='zeros')
      
    self.conv = keras.layers.Conv2D(
      filters=self.filters,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      dilation_rate=self.dilation_rate,
      activation=None,
      use_bias=False,
      kernel_initializer=self.kernel_initializer,
      kernel_regularizer=self.kernel_regularizer,
      activity_regularizer=self.activity_regularizer,
      kernel_constraint=self.kernel_constraint)
    
    self.box = keras.layers.Conv2D(
      filters=1,
      kernel_size=self.patch_size,
      strides=self.strides,
      padding=self.padding,
      activation=None,
      use_bias=False,
      kernel_initializer=keras.initializers.Constant(1 / (input_shape[3] * self.patch_size[0] * self.patch_size[1])),
      trainable=False)

    window_size = input_shape[3] * self.patch_size[0] * self.patch_size[1]
    self.variance_correction = window_size / (window_size - 1)

    if self.use_bias:
      self.bias = BiasAdd(
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint)

    if self.activation is not None:
      self.act = keras.layers.Activation(self.activation)

  def call(self, x, training=False):
    """Implement the patch norm operation using Xingtong's more efficient method.

    # todo: make more efficient for non-channel-wise runs

    """
    # (N, H', W', 1) or (N, H', W', filters) if channel_wise_nu
    means = self.box(x)
    square_means = self.box(tf.math.square(x))
    if self.use_nu:
      nu = tf.reshape(self.nu, (1, 1, 1, -1))
      stds = tf.math.sqrt((square_means - tf.math.square(means)) + tf.square(nu) + self.epsilon)
    else:
      stds = tf.math.sqrt((square_means - tf.math.square(means)) + self.epsilon)
    
    # (N, H', W', filters)
    conv = self.conv(tf.reshape(self.gamma, (1, 1, 1, -1)) * x) / stds
   
    # (1, 1, 1, filters)
    gamma_kernel_sum = tf.reduce_sum(tf.reshape(self.gamma, (1, 1, -1, 1)) * self.conv.kernel, axis=(0, 1, 2), keepdims=True)
    
    # (N, H', W', filters)
    kernel_weighted_means = means * gamma_kernel_sum / stds

    # (1, 1, 1, filters)
    beta_kernel_sum = tf.reduce_sum(tf.reshape(self.beta, (1, 1, -1, 1)) * self.conv.kernel, axis=(0, 1, 2), keepdims=True)

    # (N, H', W', filters)
    x = conv - kernel_weighted_means + beta_kernel_sum

    if self.use_bias:
      x = self.bias(x)

    if self.activation is not None:
      x = self.act(x)

    return x

  def set_weights_from_conv(self, weights):
    """Set the weights as from a convolutional layer.

    :param weights:
    :returns:
    :rtype:

    """
    self.conv.set_weights([weights[0]])
    if self.use_bias:
      self.bias.set_weights([weights[1]])

  def freeze_weights_except_norm(self):
    """Freeze the weights in this layer except for beta and gamma.
    """
    self.conv.trainable = False
    if self.use_bias:
      self.bias.trainable = False

  def unfreeze_weights(self):
    self.conv.trainable = True
    if self.use_bias:
      self.bias.trainable = True

  def set_weights_from_bn(self, weights, silent=False):
    raise NotImplementedError

  def set_weights_from_pn(self, weights):
    """Set the weights of this layer from the PatchNormConv2D layer.

    Set: 
    self.alpha = pn.beta / pn.gamma
    self.conv.kernel = pn.kernel / pn.gamma

    :param weights: 
    :returns: 
    :rtype: 

    """
    own_weights = self.get_weights()
    weights.append(own_weights[-1])
    self.set_weights(weights)

  def get_weights_for_pn(self):
    """Get the weights of this layer for the PatchNormConv2D layer.

    Get:
    pn.beta = self.alpha
    pn.gamma = 1.0
    pn.kernel = self.conv.kernel

    :param weights: 
    :returns: 
    :rtype: 

    """
    weights = self.get_weights()
    beta = weights[0]
    gamma = weights[1]
    kernel = weights[2]

    out = [beta, gamma, kernel]

    if self.use_bias:
      out.append(weights[3])

    return out


class DepthwisePatchNormConv2D(NaivePatchNormConv2D):
  """Note: currently broken"""
  
  def __init__(
      self,
      kernel_size,
      depth_multiplier=1,
      depthwise_initializer='glorot_uniform',
      depthwise_regularizer=None,
      depthwise_constraint=None,
      **kwargs):

    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = depthwise_initializer
    self.depthwise_regularizer = depthwise_regularizer
    self.depthwise_constraint = depthwise_constraint
    
    kwargs['kernel_initializer'] = depthwise_initializer
    kwargs['kernel_regularizer'] = depthwise_regularizer
    kwargs['kernel_constraint'] = depthwise_constraint

    super().__init__(1, kernel_size, **kwargs)
  
  def build(self, input_shape):
    self.beta = self.add_weight('beta',
                                shape=(input_shape[3],) if self.channel_wise else (1,),
                                dtype=self.dtype,
                                trainable=True,
                                initializer='zeros')
    self.gamma = self.add_weight('gamma',
                                 shape=(input_shape[3],) if self.channel_wise else (1,),
                                 dtype=self.dtype,
                                 trainable=True,
                                 initializer='ones')

    self.conv = keras.layers.DepthwiseConv2D(
      kernel_size=self.kernel_size,
      strides=self.kernel_size,
      padding='valid',
      depth_multiplier=self.depth_multiplier,
      activation=self.activation,
      use_bias=self.use_bias,
      depthwise_initializer=self.depthwise_initializer,
      bias_initializer=self.bias_initializer,
      depthwise_regularizer=self.depthwise_regularizer,
      bias_regularizer=self.bias_regularizer,
      activity_regularizer=self.activity_regularizer,
      depthwise_constraint=self.depthwise_constraint,
      bias_constraint=self.bias_constraint)

    # # todo: generalizer adding the conv layer over different types, etc
    # self.conv = keras.layers.DepthwiseConv2D(
    #   kernel_size=self.kernel_size,
    #   strides=self.strides,
    #   padding=self.padding,
    #   depth_multiplier=self.depth_multiplier,
    #   activation=None,
    #   use_bias=False,
    #   depthwise_initializer=self.depthwise_initializer,
    #   depthwise_regularizer=self.depthwise_regularizer,
    #   activity_regularizer=self.activity_regularizer,
    #   depthwise_constraint=self.depthwise_constraint)
    
    # self.box = keras.layers.Conv2D(
    #   filters=1,
    #   kernel_size=self.patch_size,
    #   strides=self.strides,
    #   padding=self.padding,
    #   activation=None,
    #   use_bias=False,
    #   kernel_initializer=keras.initializers.Constant(1 / (input_shape[3] * self.patch_size[0] * self.patch_size[1])),
    #   trainable=False)

    # window_size = input_shape[3] * self.patch_size[0] * self.patch_size[1]
    # self.variance_correction = window_size / (window_size - 1)

    # if self.use_bias:
    #   self.bias = BiasAdd(
    #     initializer=self.bias_initializer,
    #     regularizer=self.bias_regularizer,
    #     constraint=self.bias_constraint)

    # if self.activation is not None:
    #   self.act = keras.layers.Activation(self.activation)






