import tensorflow as tf
from tensorflow import keras

from . import utils


logger = tf.get_logger()


class PatchNormConv2D(keras.layers.Layer):
  """Notes:

  Whereas batch norm computes the mean and variance over the entire dataset, we
  compute these values over the inputs for each image patch, for a single image input.

  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               axis=3,
               **kwargs):
    """Patch norm + convolution.

    :param filters: 
    :param kernel_size: 
    :param strides: 
    :param padding: 
    :param activation: 
    :param use_bias: 
    :param kernel_initializer: 
    :param bias_initializer: 
    :param kernel_regularizer: 
    :param bias_regularizer: 
    :param activity_regularizer: 
    :param kernel_constraint: 
    :param bias_constraint: 
    :param axis: 
    :returns: 
    :rtype: 

    """
    super().__init__(**kwargs)

    self.filters = filters
    self.kernel_size = utils.tuplify(kernel_size, 2)
    self.strides = utils.tuplify(strides, 2)
    self.padding = padding
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.axis = axis

    assert self.padding == 'same' or self.kernel_size == (1, 1), 'todo: padding != same'
    assert self.axis == 3, 'todo: axis != 3'

  def build(self, input_shape):
    self.beta = self.add_weight('beta',
                                shape=(input_shape[3],),
                                dtype=tf.float32,
                                trainable=True,
                                initializer=tf.constant_initializer(0))
    self.gamma = self.add_weight('gamma',
                                 shape=(input_shape[3],),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=tf.constant_initializer(1))
    self.epsilon = tf.constant(1e-5, tf.float32)

    self.conv = keras.layers.Conv2D(
      filters=self.filters,
      kernel_size=self.kernel_size,
      strides=self.kernel_size,
      padding='valid',
      activation=self.activation,
      use_bias=self.use_bias,
      kernel_initializer=self.kernel_initializer,
      bias_initializer=self.bias_initializer,
      kernel_regularizer=self.kernel_regularizer,
      bias_regularizer=self.bias_regularizer,
      activity_regularizer=self.activity_regularizer,
      kernel_constraint=self.kernel_constraint,
      bias_constraint=self.bias_constraint)

  def call(self, x):
    # (N, H', W', h * w * C)
    patches = tf.image.extract_patches(
      images=x,
      sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
      strides=[1, self.strides[0], self.strides[1], 1],
      rates=[1, 1, 1, 1],
      padding=self.padding.upper())    

    # (N, H', W', h, w, C)
    patches = tf.reshape(patches, (-1, patches.shape[1], patches.shape[2], self.kernel_size[0], self.kernel_size[1], x.shape[3]))
    
    # (N, H', W', 1, 1, 1)
    mus = tf.math.reduce_mean(patches, axis=3, keepdims=True)
    sigs = tf.math.reduce_std(patches, axis=3, keepdims=True)

    # (N, H', W', 1, 1, 1)
    centered = (patches - mus) / tf.sqrt(tf.square(sigs) + self.epsilon)
    shifted = tf.reshape(self.gamma, (1, 1, 1, 1, 1, -1)) * centered + tf.reshape(self.beta, (1, 1, 1, 1, 1, -1))

    # (N, H', h, W', w, C)
    shifted = tf.transpose(shifted, perm=[0, 1, 3, 2, 4, 5])
    shifted = tf.reshape(shifted, [-1, shifted.shape[1] * self.kernel_size[0], shifted.shape[2] * self.kernel_size[1], x.shape[3]])
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
                   'axis': self.axis})
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
    
    
class EfficientPatchNormConv2D(PatchNormConv2D):
  def build(self, input_shape):
      # TODO: Both pairs of beta and gamma setting work fine. (seems like filters one converge a bit faster?)
    # self.beta = self.add_weight(
    #   'beta',
    #   shape=(self.filters),
    #   dtype=self.dtype,
    #   trainable=True,
    #   initializer=tf.constant_initializer(0))
    # self.gamma = self.add_weight(
    #   'gamma',
    #   shape=(self.filters),
    #   dtype=self.dtype,
    #   trainable=True,
    #   initializer=tf.constant_initializer(1))
    self.alpha = self.add_weight(
      'alpha',
      shape=(input_shape[3],),
      dtype=self.dtype,
      trainable=True,
      initializer=tf.constant_initializer(0))  # sort of like a bias, gets multiplied along the in_channels dimension of the conv kernel
    self.epsilon = tf.constant(1e-5, tf.float32)

    self.conv = keras.layers.Conv2D(
      filters=self.filters,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      activation=None,
      use_bias=False,
      kernel_initializer=self.kernel_initializer,
      kernel_regularizer=self.kernel_regularizer,
      activity_regularizer=self.activity_regularizer,
      kernel_constraint=self.kernel_constraint,
      dtype=self.dtype)

    if self.use_bias:
      self.bias = BiasAdd(
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        dtype=self.dtype)

    if self.activation is not None:
      self.act = keras.layers.Activation(self.activation)

    self.window = tf.ones((self.kernel_size[0], self.kernel_size[1], input_shape[3], 1), dtype=tf.float32) / (input_shape[3] * self.kernel_size[0] * self.kernel_size[1])
    self.variance_correction = 2.0 / (input_shape[3] * self.kernel_size[0] * self.kernel_size[1])

  def call(self, x):
    """Implement the patch norm operation using Xingtong's more efficient method.

    """
    # (N, H', W', 1)
    means = tf.nn.conv2d(x, self.window, strides=self.strides, padding=self.padding.upper())
    square_means = tf.nn.conv2d(tf.math.square(x), self.window, strides=self.strides, padding=self.padding.upper())
    stds = tf.math.sqrt(square_means + tf.math.square(means) * self.variance_correction + self.epsilon)
    # std_map = tf.math.sqrt(self.var_bias_correction * (mean_sq_map - tf.math.square(mean_map)) + self.epsilon)

    # (N, H', W', filters)
    conv = self.conv(x)

    # (1, 1, 1, filters)
    kernel_sum = tf.reduce_sum(self.conv.kernel, axis=(0, 1, 2), keepdims=True)
    weighted_kernel = self.conv.kernel * tf.reshape(self.alpha, (1, 1, -1, 1))
    weighted_kernel_sum = tf.reduce_sum(weighted_kernel, axis=(0, 1, 2), keepdims=True)

    x = (conv - means * kernel_sum) / stds + weighted_kernel_sum
    
    # Xingtong:
    # x = (x - (mean_map * kernel_summation)) / std_map
    # x = self.gamma * x + self.beta * kernel_summation

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
